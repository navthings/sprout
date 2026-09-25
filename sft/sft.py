import argparse
import json
import math
import os
import shutil
import subprocess
import time
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from tokenizers import Tokenizer, pre_tokenizers, processors

HERE = Path(__file__).parent
EOS = 2  # </s> ends every assistant turn and, as in pretraining, starts every sequence
BF16 = mx.bfloat16
THROTTLE = 0.8  # fanless air: budget for ~20% slower than the cold-start speed measured at the beginning
STRIP = ("<s>", "</s>", "<unk>", "[INST]", "[/INST]", "<<SYS>>", "<</SYS>>")
# same format as chat_pieces, for transformers / mlx-lm / llama.cpp; the ollama version is in gguf/Modelfile.chat
CHAT_TEMPLATE = (
    "{{- bos_token }}"
    "{%- set msgs = messages %}"
    "{%- if msgs[0]['role'] == 'system' %}{{ '<<SYS>>\\n' + msgs[0]['content'] | trim + '\\n<</SYS>>\\n\\n' }}"
    "{%- set msgs = msgs[1:] %}{%- endif %}"
    "{%- for m in msgs %}"
    "{%- if m['role'] == 'user' %}{{ '[INST] ' + m['content'] | trim + ' [/INST]' }}"
    "{%- elif m['role'] == 'assistant' %}{{ ' ' + m['content'] | trim + eos_token }}{%- endif %}"
    "{%- endfor %}"
)
SAMPLE_PROMPTS = ["What causes the seasons on Earth?", "Write a haiku about a cat who hates Mondays."]


class Linear(nn.Module):
    # fp32 master weights for the optimizer; matmuls run in the activation dtype
    def __init__(self, d_in, d_out):
        super().__init__()
        self.weight = mx.zeros((d_out, d_in))

    def __call__(self, x):
        return x @ self.weight.astype(x.dtype).T


class RMSNorm(nn.Module):
    def __init__(self, d, eps):
        super().__init__()
        self.weight = mx.ones((d,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, self.weight.astype(x.dtype), self.eps)


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        d, self.hd = c["hidden_size"], c["head_dim"]
        self.nh, self.nkv = c["num_attention_heads"], c["num_key_value_heads"]
        self.theta = c.get("rope_theta") or c.get("rope_parameters", {}).get("rope_theta", 10000.0)
        self.q_proj = Linear(d, self.nh * self.hd)
        self.k_proj = Linear(d, self.nkv * self.hd)
        self.v_proj = Linear(d, self.nkv * self.hd)
        self.o_proj = Linear(self.nh * self.hd, d)

    def __call__(self, x, mask, cache):
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.nh, self.hd).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.nkv, self.hd).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.nkv, self.hd).transpose(0, 2, 1, 3)
        offset = 0 if cache is None else cache[0].shape[2]
        # hf layout (rotate_half), hence traditional=False
        q = mx.fast.rope(q, self.hd, traditional=False, base=self.theta, scale=1.0, offset=offset)
        k = mx.fast.rope(k, self.hd, traditional=False, base=self.theta, scale=1.0, offset=offset)
        if cache is not None:
            k, v = mx.concatenate([cache[0], k], axis=2), mx.concatenate([cache[1], v], axis=2)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.hd**-0.5, mask=mask)
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(B, L, -1)), (k, v)


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        d, f = c["hidden_size"], c["intermediate_size"]
        self.gate_proj, self.up_proj, self.down_proj = Linear(d, f), Linear(d, f), Linear(f, d)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.self_attn, self.mlp = Attention(c), MLP(c)
        self.input_layernorm = RMSNorm(c["hidden_size"], c["rms_norm_eps"])
        self.post_attention_layernorm = RMSNorm(c["hidden_size"], c["rms_norm_eps"])

    def __call__(self, x, mask, cache):
        h, cache = self.self_attn(self.input_layernorm(x), mask, cache)
        x = x + h
        return x + self.mlp(self.post_attention_layernorm(x)), cache


class Body(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.embed_tokens = nn.Embedding(c["vocab_size"], c["hidden_size"])
        self.layers = [Block(c) for _ in range(c["num_hidden_layers"])]
        self.norm = RMSNorm(c["hidden_size"], c["rms_norm_eps"])


class Llama(nn.Module):
    # parameter names match transformers' LlamaForCausalLM, so hf safetensors load and save as-is
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Body(config)

    def __call__(self, ids, mask=None, cache=None):
        m = self.model
        emb = m.embed_tokens.weight
        h = emb[ids].astype(BF16)
        cache = cache or [None] * len(m.layers)
        new_cache = []
        for layer, c in zip(m.layers, cache):
            h, c = layer(h, mask, c)
            new_cache.append(c)
        return m.norm(h) @ emb.astype(BF16).T, new_cache  # tied embeddings


def load_model(folder):
    folder = Path(folder)
    model = Llama(json.loads((folder / "config.json").read_text()))
    model.load_weights(str(folder / "model.safetensors"))
    mx.eval(model.parameters())
    return model


def load_tokenizer(folder):
    # llama.cpp puts a space prefix on every text run after a special token; "always" makes hf agree,
    # so `[INST]` after `</s>` tokenizes the same in training, ollama and transformers
    tok = Tokenizer.from_file(str(Path(folder) / "tokenizer.json"))
    tok.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always", split=False)
    tok.post_processor = processors.TemplateProcessing(single="</s> $A", special_tokens=[("</s>", EOS)])
    return tok


def clean(text):
    for s in STRIP:
        text = text.replace(s, "")
    return text.strip()


def chat_pieces(messages):
    """[(text, is_assistant)] for [system] (user assistant)* [user]; a trailing user turn ends at [/INST]"""
    pieces, prefix = [], ""
    if messages and messages[0]["role"] == "system":
        prefix = f"<<SYS>>\n{clean(messages[0]['content'])}\n<</SYS>>\n\n"
        messages = messages[1:]
    for i, m in enumerate(messages):
        role = ("user", "assistant")[i % 2]
        if m["role"] != role:
            raise ValueError(f"message {i} should be {role}, got {m['role']}")
        if not clean(m["content"]):
            raise ValueError(f"message {i} is empty")
        if role == "user":
            pieces.append((f"{prefix}[INST] {clean(m['content'])} [/INST]", False))
            prefix = ""
        else:
            pieces.append((" " + clean(m["content"]), True))
    return pieces


def assemble(pieces, encoded, limit=None):
    """ids and loss mask; assistant turns get </s> appended and trained on. stops before a turn that won't fit"""
    ids, mask = [EOS], [0]
    for i in range(0, len(pieces), 2):
        turn_ids, turn_mask = [], []
        for (_, is_asst), enc in zip(pieces[i:i + 2], encoded[i:i + 2]):
            extra = [EOS] if is_asst else []
            turn_ids += enc + extra
            turn_mask += [int(is_asst)] * (len(enc) + len(extra))
        if limit and len(ids) + len(turn_ids) > limit:
            break
        ids += turn_ids
        mask += turn_mask
    return ids, mask


def encode_chat(tok, messages):
    pieces = chat_pieces(messages)
    return assemble(pieces, [e.ids for e in tok.encode_batch([p for p, _ in pieces], add_special_tokens=False)])[0]


def prepare(tok, name, split, seq, cache_dir):
    """tokenize a chat dataset once into flat uint16 tokens + loss mask, cached as .npz"""
    path = cache_dir / f"{name.replace('/', '--')}_{split}_{seq}.npz"
    if path.exists():
        d = np.load(path)
        return d["tokens"], d["mask"], d["starts"]

    from datasets import load_dataset

    ds = load_dataset(name, split=split)
    tokens, masks, lengths, skipped = [], [], [], 0
    for batch in ds.iter(batch_size=20_000):
        convs = []
        for msgs in batch["messages"]:
            try:
                p = chat_pieces(msgs)
            except ValueError:
                p = []
            if len(p) < 2:
                skipped += 1
                continue
            convs.append(p)
        flat = [e.ids for e in tok.encode_batch([t for p in convs for t, _ in p], add_special_tokens=False)]
        k = 0
        for p in convs:
            ids, mask = assemble(p, flat[k:k + len(p)], limit=seq + 1)
            k += len(p)
            if not any(mask):  # even the first turn was too long
                skipped += 1
                continue
            tokens.append(np.array(ids, np.uint16))
            masks.append(np.array(mask, np.uint8))
            lengths.append(len(ids))
        print(f"  {split}: {len(lengths):,} conversations tokenized, {skipped:,} skipped", flush=True)

    starts = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    tokens, masks = np.concatenate(tokens), np.concatenate(masks)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(path, tokens=tokens, mask=masks, starts=starts)
    print(f"  {split}: {len(tokens) / 1e6:.1f}M tokens, {masks.mean():.0%} assistant -> {path.name}")
    return tokens, masks, starts


def pack(lengths, seq, rng, open_max=32):
    """greedy-fill shuffled conversations into rows of `seq` positions (a conversation of n tokens needs n-1)"""
    rows, open_rows, room = [], [], []
    for i in rng.permutation(len(lengths)):
        n = int(lengths[i]) - 1
        for j, r in enumerate(room):
            if n <= r:
                open_rows[j].append(i)
                room[j] -= n
                if room[j] < 16:
                    rows.append(open_rows.pop(j))
                    room.pop(j)
                break
        else:
            open_rows.append([i])
            room.append(seq - n)
            if len(open_rows) > open_max:
                rows.append(open_rows.pop(0))
                room.pop(0)
    rows += open_rows
    rng.shuffle(rows)
    return rows


def make_batch(data, rows, seq):
    tokens, mask, starts = data
    x = np.zeros((len(rows), seq), np.int32)
    y = np.zeros((len(rows), seq), np.int32)
    w = np.zeros((len(rows), seq), np.float32)
    seg = np.full((len(rows), seq), -1, np.int32)
    for b, row in enumerate(rows):
        p = 0
        for k, i in enumerate(row):
            s, e = starts[i], starts[i + 1]
            n = e - s - 1
            x[b, p:p + n], y[b, p:p + n] = tokens[s:e - 1], tokens[s + 1:e]
            w[b, p:p + n], seg[b, p:p + n] = mask[s + 1:e], k
            p += n
    return mx.array(x), mx.array(y), mx.array(w), mx.array(seg)


def loss_fn(model, x, y, w, seg):
    """summed cross-entropy over assistant tokens; packed conversations can't see each other"""
    L = x.shape[1]
    causal = mx.tril(mx.ones((L, L), dtype=mx.bool_))
    mask = ((seg[:, :, None] == seg[:, None, :]) & causal)[:, None]
    logits, _ = model(x, mask)
    ce = nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="none")
    return (ce * w).sum()


def generate(model, ids, max_new=300, temp=0.7, top_k=40, rep_penalty=1.1):
    """yields token ids until </s>, max_new, or the 1024-token context runs out"""
    ctx = model.config["max_position_embeddings"]
    seen = set(ids)
    x, cache = mx.array(ids)[None], None
    for _ in range(min(max_new, ctx - len(ids))):
        logits, cache = model(x, "causal" if cache is None else None, cache)
        logits = logits[0, -1].astype(mx.float32)
        if rep_penalty != 1.0:
            idx = mx.array(sorted(seen))
            l = logits[idx]
            logits[idx] = mx.where(l > 0, l / rep_penalty, l * rep_penalty)
        if temp == 0:
            nxt = mx.argmax(logits)
        else:
            if top_k:
                logits = mx.where(logits < mx.topk(logits, top_k).min(), -mx.inf, logits)
            nxt = mx.random.categorical(logits / temp)
        nxt = nxt.item()
        if nxt == EOS:
            return
        seen.add(nxt)
        yield nxt
        x = mx.array([[nxt]])


def export(model, tok, base, out):
    """hf LlamaForCausalLM folder with the chat template; convert to gguf with llama.cpp"""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out / "model.safetensors"), dict(tree_flatten(model.parameters())), metadata={"format": "pt"})
    shutil.copy(Path(base) / "config.json", out / "config.json")
    tok.save(str(out / "tokenizer.json"))
    (out / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "PreTrainedTokenizerFast", "bos_token": "</s>", "eos_token": "</s>", "unk_token": "<unk>",
        "add_bos_token": True, "add_eos_token": False, "clean_up_tokenization_spaces": False, "model_max_length": 1024,
        "chat_template": CHAT_TEMPLATE,
    }, indent=2))
    print(f"exported -> {out}")


def save_ckpt(ckpt, model, optimizer, meta):
    ckpt.mkdir(parents=True, exist_ok=True)
    # write then rename, so a crash mid-save never leaves a half-written checkpoint
    for name, tree in (("model", model.parameters()), ("optim", optimizer.state)):
        mx.save_safetensors(str(ckpt / f"{name}.tmp.safetensors"), dict(tree_flatten(tree)))
    for name in ("model", "optim"):
        (ckpt / f"{name}.tmp.safetensors").replace(ckpt / f"{name}.safetensors")
    (ckpt / "meta.json").write_text(json.dumps(meta, indent=2))


def show_samples(model, tok, n=80):
    for prompt in SAMPLE_PROMPTS:
        ids = encode_chat(tok, [{"role": "user", "content": prompt}])
        reply = tok.decode(list(generate(model, ids, max_new=n, temp=0)))
        print(f"  > {prompt}\n    {reply!r}")


def main():
    ap = argparse.ArgumentParser(description="chat sft on apple silicon (mlx)")
    ap.add_argument("--base", default=str(HERE.parent.parent / "lilbase" / "mac" / "hf"), help="hf folder of the base model")
    ap.add_argument("--out", default=str(HERE / "hf-chat"))
    ap.add_argument("--ckpt", default=str(HERE / "sft_ckpt"))
    ap.add_argument("--data", default="HuggingFaceTB/smol-smoltalk")
    ap.add_argument("--hours", type=float, default=12, help="wall-clock budget; sets the step count after measuring speed")
    ap.add_argument("--steps", type=int, help="fixed step count instead of --hours")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--micro", type=int, default=2, help="sequences per forward pass; 4 needs more than 16GB")
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--save-mins", type=float, default=30)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fresh", action="store_true", help="ignore an existing checkpoint")
    ap.add_argument("--bench", action="store_true", help="time 30 steps, print speed, memory and the budget, then exit")
    ap.add_argument("--export-only", action="store_true", help="export the latest checkpoint and exit")
    args = ap.parse_args()

    ckpt, per_step = Path(args.ckpt), args.micro * args.accum
    meta_path = ckpt / "meta.json"
    resume = meta_path.exists() and not args.fresh and not args.bench
    meta = json.loads(meta_path.read_text()) if resume else {}

    tok = load_tokenizer(args.base)
    model = load_model(args.base)
    seq = model.config["max_position_embeddings"]
    if resume:
        model.load_weights(str(ckpt / "model.safetensors"))
        if (meta["micro"], meta["accum"], meta["seed"], meta["data"]) != (args.micro, args.accum, args.seed, args.data):
            raise SystemExit(f"checkpoint was made with micro/accum/seed/data {meta['micro']}/{meta['accum']}/"
                             f"{meta['seed']}/{meta['data']}; pass the same or use --fresh")
    if args.export_only:
        return export(model, tok, args.base, args.out)

    if shutil.which("caffeinate"):  # keep the mac awake (not with the lid shut) until we exit
        subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])

    print("data:")
    train = prepare(tok, args.data, "train", seq, HERE / "sft_data")
    val = prepare(tok, args.data, "test", seq, HERE / "sft_data")
    rows = pack(np.diff(train[2]), seq, np.random.default_rng(args.seed))
    val_rows = pack(np.diff(val[2]), seq, np.random.default_rng(0))[:16 * args.micro]
    data_steps = len(rows) // per_step
    fill = np.diff(train[2]).sum() / (len(rows) * (seq + 1))
    print(f"  {len(rows):,} packed rows ({fill:.0%} full) = {data_steps:,} steps of {per_step * seq:,} tokens")

    def meta_now():
        return dict(step=step, total_steps=total_steps, micro=args.micro, accum=args.accum, seed=args.seed, data=args.data)

    optimizer = optim.AdamW(learning_rate=0.0, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    optimizer.init(model.trainable_parameters())
    if resume:
        optimizer.state = tree_unflatten(list(mx.load(str(ckpt / "optim.safetensors")).items()))

    # compiled functions capture these state trees, so they're built after any checkpoint is loaded
    grad_fn = nn.value_and_grad(model, loss_fn)
    state = [model.state, optimizer.state]

    # value_and_grad swaps traced params into the model, so its state is an output too
    @partial(mx.compile, inputs=model.state, outputs=model.state)
    def micro_step(x, y, w, seg):
        return grad_fn(model, x, y, w, seg)

    @partial(mx.compile, inputs=state, outputs=state)
    def apply(grads, scale):
        grads, norm = optim.clip_grad_norm(tree_map(lambda g: g * scale, grads), 1.0)
        optimizer.update(model, grads)
        return norm

    @partial(mx.compile, inputs=model.state)
    def eval_step(x, y, w, seg):
        return loss_fn(model, x, y, w, seg)

    def evaluate():
        total = n = 0.0
        for i in range(0, len(val_rows), args.micro):
            b = make_batch(val, val_rows[i:i + args.micro], seq)
            total += eval_step(*b).item()
            n += b[2].sum().item()
        return total / n

    total_steps = meta.get("total_steps") or args.steps or (30 if args.bench else None)
    step = meta.get("step", 0)
    lr_at = lambda s: args.lr * (s + 1) / args.warmup if s < args.warmup else (
        args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, (s - args.warmup) / max(1, (total_steps or data_steps) - args.warmup))))))

    print(f"training from step {step}" + (f" of {total_steps}" if total_steps else ", step count set after measuring speed"))
    mx.reset_peak_memory()
    t_last_save = t_log = time.time()
    loss_acc = tok_acc = 0.0
    step_times = []
    try:
        while step < (total_steps or data_steps):
            t0 = time.time()
            optimizer.learning_rate = lr_at(step)
            grads, loss_sum, n_tok = None, 0.0, 0.0
            for m in range(args.accum):
                lo = (step * args.accum + m) * args.micro
                x, y, w, seg = make_batch(train, rows[lo:lo + args.micro], seq)
                loss, g = micro_step(x, y, w, seg)
                grads = g if grads is None else tree_map(mx.add, grads, g)
                loss_sum, n_tok = loss_sum + loss, n_tok + w.sum()
                mx.eval(grads, loss_sum, n_tok)
            norm = apply(grads, 1.0 / n_tok)
            mx.eval(model.parameters(), optimizer.state, norm)
            step += 1
            loss_acc += loss_sum.item()
            tok_acc += n_tok.item()
            step_times.append(time.time() - t0)

            if total_steps is None and step == 30:  # early steps include compilation and memory settling
                sec = float(np.mean(step_times[10:]))
                total_steps = min(data_steps, int(args.hours * 3600 * THROTTLE / sec))
                print(f"  {per_step * seq / sec:,.0f} tok/s ({sec:.2f}s/step) -> {total_steps:,} steps "
                      f"(~{total_steps * sec / 3600:.1f}h at this speed, budgeted for {args.hours:g}h)")

            if step % 10 == 0 or step == total_steps:
                sec = float(np.mean(step_times[-10:]))
                eta = f"{(total_steps - step) * sec / 3600:.1f}h" if total_steps else "?"
                print(f"step {step:5d}/{total_steps or '?'}  loss {loss_acc / tok_acc:.4f}  lr {lr_at(step - 1):.2e}  "
                      f"norm {norm.item():.2f}  {per_step * seq / sec:,.0f} tok/s  "
                      f"mem {mx.get_peak_memory() / 1e9:.1f}GB  eta {eta}", flush=True)
                loss_acc = tok_acc = 0.0

            if args.bench:
                continue
            if step % args.eval_every == 0 or step == total_steps:
                print(f"  eval: val loss {evaluate():.4f}")
                show_samples(model, tok)
            if time.time() - t_last_save > args.save_mins * 60 and step < total_steps:
                save_ckpt(ckpt, model, optimizer, meta_now())
                t_last_save = time.time()
                print(f"  saved checkpoint at step {step}")
    except KeyboardInterrupt:
        if args.bench or total_steps is None:
            raise SystemExit("\nstopped")
        save_ckpt(ckpt, model, optimizer, meta_now())
        raise SystemExit(f"\nstopped, checkpoint saved at step {step}. run the same command to resume, "
                         f"or --export-only to export it as is")

    if args.bench:
        sec = float(np.mean(step_times[5:]))
        budget = int(args.hours * 3600 * THROTTLE / sec)
        print(f"\n{per_step * seq / sec:,.0f} tok/s, peak memory {mx.get_peak_memory() / 1e9:.1f}GB. "
              f"--hours {args.hours:g} gives {min(budget, data_steps):,} steps "
              f"({min(budget, data_steps) * per_step * seq / 1e6:.0f}M tokens, data has {data_steps:,})")
        return

    export(model, tok, args.base, args.out)
    save_ckpt(ckpt, model, optimizer, meta_now())


if __name__ == "__main__":
    main()
