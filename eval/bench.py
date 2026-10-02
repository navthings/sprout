import json
import math
import os
import re
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


HERE = os.path.dirname(os.path.abspath(__file__))

MODELS = {
    "gpt2": "openai-community/gpt2",
    "lilbase": "navthings/lilbase",
    "gpt2-medium": "openai-community/gpt2-medium",
    "sprout-chat": os.path.expanduser("~/Models/sprout/sprout-chat"),
    "gpt2-large": "openai-community/gpt2-large",
    "gpt2-xl": "openai-community/gpt2-xl",
}
OUT = os.path.join(HERE, "results.json")

N_HELLA = 2000
N_LAMBADA = 2000
N_DOCS = 300
CTX = 1024

dev = "mps" if torch.backends.mps.is_available() else "cpu"


def hella_clean(s):
    s = s.strip()
    s = s.replace("[title]", ". ")
    s = re.sub(r"\[[^\]]*\]", "", s)
    return s


def load_tasks():
    print("Loading datasets...", flush=True)

    lambada = load_dataset(
        "EleutherAI/lambada_openai",
        split="test",
    )

    lambada_items = []
    for row in lambada:
        text = row["text"].strip()
        if " " not in text:
            continue
        ctx, cont = text.rsplit(" ", 1)
        lambada_items.append((ctx, " " + cont))

    hellaswag = load_dataset(
        "Rowan/hellaswag",
        split="validation",
        revision="refs/convert/parquet",
    )

    hella_items = []
    for row in hellaswag.select(range(min(N_HELLA, len(hellaswag)))):
        ctx = hella_clean(row["ctx"])
        # endings need a leading space, ctx ends mid-sentence ("... he" + "is using")
        choices = [" " + x.strip() for x in row["endings"]]
        label = int(row["label"])
        hella_items.append((ctx, choices, label))

    arc = load_dataset(
        "allenai/ai2_arc",
        "ARC-Easy",
        split="test",
    )

    arc_items = []
    for row in arc:
        labels = row["choices"]["label"]
        texts = row["choices"]["text"]
        choices = [f" {x}" for x in texts]

        answer = row["answerKey"]
        if answer not in labels:
            continue

        label = labels.index(answer)
        arc_items.append((row["question"], choices, label))

    fineweb = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        "sample-10BT",
        split="train",
        streaming=True,
    )

    fineweb_items = []
    for row in fineweb:
        fineweb_items.append(row["text"])
        if len(fineweb_items) >= N_DOCS:
            break

    wikitext = load_dataset(
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
        split="test",
    )

    wiki_items = [
        row["text"]
        for row in wikitext
        if row["text"].strip()
    ]

    return {
        "lambada": lambada_items[:N_LAMBADA],
        "hellaswag": hella_items,
        "arc_easy": arc_items,
        "fineweb_edu": fineweb_items,
        "wikitext": wiki_items,
    }


class Scorer:
    def __init__(self, path):
        print(f"Loading {path} on {dev}...", flush=True)

        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            dtype=torch.float32,
        ).to(dev).eval()

        self.eos = self.tok.eos_token_id
        self.head = self.model.get_output_embeddings()

    def enc(self, s):
        return self.tok.encode(
            s,
            add_special_tokens=False,
        )

    def pair(self, ctx, cont):
        whole = self.enc(ctx + cont)
        c = self.enc(ctx)

        if whole[:len(c)] == c:
            cont_ids = whole[len(c):]
        else:
            cont_ids = self.enc(cont)

        ids = ([self.eos] + c + cont_ids)[-CTX:]

        return ids, len(cont_ids)

    @torch.inference_mode()
    def token_lls(self, seqs, spans):
        # pad to a multiple of 32, not 128: short questions were mostly padding
        n = -(-max(len(s) for s in seqs) // 32) * 32
        x = torch.zeros(len(seqs), n, dtype=torch.long)
        mask = torch.zeros(len(seqs), n, dtype=torch.long)
        for i, s in enumerate(seqs):
            x[i, :len(s)] = torch.tensor(s)
            mask[i, :len(s)] = 1

        h = self.model.base_model(x.to(dev), attention_mask=mask.to(dev)).last_hidden_state

        # only run the vocab projection where an answer token is being predicted
        rows, pos, tgt, counts = [], [], [], []
        for i, (a, b) in enumerate(spans):
            for p in range(a - 1, b - 1):
                rows.append(i)
                pos.append(p)
                tgt.append(seqs[i][p + 1])
            counts.append(b - a)
        t = torch.tensor(tgt, device=dev)
        lp = self.head(h[torch.tensor(rows, device=dev), torch.tensor(pos, device=dev)]).float().log_softmax(-1)
        tok_lp = lp.gather(-1, t[:, None]).squeeze(-1).cpu()
        hit = (lp.argmax(-1) == t).cpu()

        out, k = [], 0
        for c in counts:
            out.append((tok_lp[k:k + c].sum().item(), bool(hit[k:k + c].all())))
            k += c
        return out

    def tick(self, i, n, what):
        if i % 400 == 0:
            print(f"    {what} {i}/{n}", flush=True)

    def score(self, pairs):
        seqs, lens = zip(
            *(self.pair(c, t) for c, t in pairs)
        )

        return self.token_lls(
            list(seqs),
            [
                (len(s) - k, len(s))
                for s, k in zip(seqs, lens)
            ],
        )

    def choice_acc(self, items, bs):
        hits = 0

        for i in range(0, len(items), bs):
            self.tick(i, len(items), "choices")
            group = items[i:i + bs]

            lls = iter(
                ll
                for ll, _ in self.score(
                    (ctx, c)
                    for ctx, choices, _ in group
                    for c in choices
                )
            )

            for ctx, choices, label in group:
                norm = [
                    next(lls) / len(c)
                    for c in choices
                ]

                hits += int(
                    max(
                        range(len(norm)),
                        key=norm.__getitem__,
                    ) == label
                )

        return hits / len(items)

    def lambada(self, items, bs):
        hits, nll, ntok = 0, 0.0, 0

        for i in range(0, len(items), bs):
            self.tick(i, len(items), "lambada")
            for ll, g in self.score(items[i:i + bs]):
                hits += g
                nll -= ll
                ntok += 1

        return (
            hits / len(items),
            math.exp(nll / ntok),
        )

    def bpb(self, items, bs=4):
        total_ll = 0.0
        total_bytes = 0

        for i in range(0, len(items), bs):
            pairs = []

            for text in items[i:i + bs]:
                # same 2000-char cut for every model so long docs never overflow the 1024 window
                text = text.strip()[:2000]
                if not text:
                    continue

                pairs.append(("", text))

            if not pairs:
                continue

            scores = self.score(pairs)

            for (ll, _), (_, text) in zip(scores, pairs):
                total_ll += ll
                total_bytes += len(text.encode("utf-8"))

        return -total_ll / math.log(2) / total_bytes


if __name__ == "__main__":
    tasks = load_tasks()

    results = json.load(open(OUT)) if os.path.exists(OUT) else {}

    for name, path in MODELS.items():
        done = results.get(name, {})
        if "wikitext_bpb" in done and done.get("hella_v2"):
            print(f"skip {name}, already done", flush=True)
            continue
        if "wikitext_bpb" in done:
            # only hellaswag changed (spacing fix), the other scores still stand
            print(f"\nMODEL: {name} (hellaswag rerun)", flush=True)
            scorer = Scorer(path)
            done["hellaswag_acc_norm"] = scorer.choice_acc(tasks["hellaswag"], 8)
            done["hella_v2"] = True
            print(f"  hellaswag={done['hellaswag_acc_norm']:.4f}", flush=True)
            json.dump(results, open(OUT, "w"), indent=2)
            del scorer
            if dev == "mps":
                torch.mps.empty_cache()
            continue

        print(f"\nMODEL: {name}", flush=True)
        t0 = time.time()
        scorer = Scorer(path)
        r = {"params_m": round(sum(p.numel() for p in scorer.model.parameters()) / 1e6, 1)}

        r["lambada_acc"], r["lambada_ppl"] = scorer.lambada(tasks["lambada"], 16)
        print(f"  lambada acc={r['lambada_acc']:.4f} ppl={r['lambada_ppl']:.2f} ({time.time()-t0:.0f}s)", flush=True)
        r["hellaswag_acc_norm"] = scorer.choice_acc(tasks["hellaswag"], 8)
        print(f"  hellaswag={r['hellaswag_acc_norm']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        r["arc_easy_acc_norm"] = scorer.choice_acc(tasks["arc_easy"], 8)
        print(f"  arc_easy={r['arc_easy_acc_norm']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        r["fineweb_edu_bpb"] = scorer.bpb(tasks["fineweb_edu"], 4)
        print(f"  fineweb_edu bpb={r['fineweb_edu_bpb']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        r["wikitext_bpb"] = scorer.bpb(tasks["wikitext"], 4)
        print(f"  wikitext bpb={r['wikitext_bpb']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        r["seconds"] = round(time.time() - t0)
        r["hella_v2"] = True

        # save after every model so a crash or throttle kill keeps what's done
        results[name] = r
        json.dump(results, open(OUT, "w"), indent=2)

        del scorer
        if dev == "mps":
            torch.mps.empty_cache()

    print(json.dumps(results, indent=2))
