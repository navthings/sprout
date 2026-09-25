# %% [markdown]
# # lilbase2
#
# 523M llama-style model, 12B tokens, kaggle tpu v5e-8. one session gets through ~8.4h of it, then saves and stops.
#
# **first run:** settings > accelerator **TPU v5e-8**, internet **on**. optional: add an `HF_TOKEN` secret (add-ons > secrets) to dodge hf rate limits. then **save version > save & run all**.
#
# **every run after that:** open the editor, **add input > your work > this notebook** (or update the existing input to the newest version), then **save version** again. it finds the newest `lilbase2_step_*` checkpoint in `/kaggle/working` or anywhere under `/kaggle/input` on its own and carries on from there. data position, optimizer state, lr schedule and loss history all come with it.
#
# to resume from one specific checkpoint instead, set `RESUME_FROM` in the config cell.

# %%
import os, warnings, logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # has to be set before jax imports
os.environ.setdefault("TPU_STDERR_LOG_LEVEL", "3")
os.environ.setdefault("TPU_MIN_LOG_LEVEL", "3")
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["HF_HUB_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import importlib.metadata, importlib.util, subprocess, sys

# pinned so a saved stream position means the same thing in every session
DATASETS_VERSION = "5.0.1"


def installed(pkg):
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return None


need = [p for p in ("safetensors", "transformers", "psutil", "matplotlib") if importlib.util.find_spec(p) is None]
if installed("datasets") != DATASETS_VERSION:
    need.append(f"datasets=={DATASETS_VERSION}")
if need:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *need], check=True)

import glob, inspect, json, math, queue, re, shutil, threading, time
from functools import partial

RUN_START = time.time()

# hf token avoids rate limits over a 9h stream (ask me how i know)
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF_TOKEN loaded from kaggle secrets")
except Exception as e:
    print(f"no HF_TOKEN secret ({type(e).__name__}), streaming unauthenticated")

import datasets, transformers
import jax
import jax.numpy as jnp
import numpy as np
import psutil
from datasets import Features, Value, interleave_datasets, load_dataset
from safetensors.numpy import load_file, save_file
from transformers import AutoTokenizer

DEVICES = jax.devices()
N_CHIPS = len(DEVICES)
DEV = DEVICES[0]
print(f"jax {jax.__version__}, datasets {datasets.__version__}, {DEV.platform} {DEV.device_kind}, {N_CHIPS} device(s)")
print(f"host RAM {psutil.virtual_memory().total / 2**30:.0f} GB")

# kaggle sometimes hands out 1 chip instead of 8
if DEV.platform != "tpu" and not os.environ.get("LILBASE_ALLOW_CPU"):
    raise RuntimeError("no tpu found, check the accelerator is set to TPU v5e-8")
if DEV.platform == "tpu" and N_CHIPS != 8:
    raise RuntimeError(f"got {N_CHIPS} chips instead of 8. restart the session (sometimes factory reset) and try again")
if datasets.__version__ != DATASETS_VERSION:
    raise RuntimeError(f"datasets {datasets.__version__} is loaded but {DATASETS_VERSION} is pinned. restart the session so the install takes")

mesh = jax.sharding.Mesh(np.array(DEVICES), ("data",))
P = jax.sharding.PartitionSpec
repl = jax.sharding.NamedSharding(mesh, P())
sharded = jax.sharding.NamedSharding(mesh, P("data"))

# %%
VOCAB, D, N_LAYERS, N_HEADS, N_KV, D_FF = 32000, 1280, 28, 20, 4, 3456
HEAD_DIM = D // N_HEADS
ROPE_THETA, NORM_EPS = 10000.0, 1e-5

SEQ = 1024
BS, GRAD_ACC = 4, 16
TOTAL_TOKENS = 12_000_000_000
SAVE_EVERY, PRINT_EVERY, STATS_EVERY, EVAL_EVERY = 500, 50, 250, 500

# wsd schedule: linear warmup, flat, then linear decay over the last DECAY_FRAC of steps.
# TOTAL_TOKENS can be changed between sessions as long as the decay hasnt started yet
LR, WARMUP_STEPS, DECAY_FRAC, MIN_LR_RATIO = 3e-4, 1500, 0.15, 0.1
BETA1, BETA2, ADAM_EPS, WEIGHT_DECAY, GRAD_CLIP = 0.9, 0.95, 1e-8, 0.1, 1.0

# (name, hf path, config, split, text column, share of training tokens)
SOURCES = [
    ("fineweb-edu", "HuggingFaceFW/fineweb-edu", "sample-100BT", "train", "text", 0.55),
    ("dclm", "mlfoundations/dclm-baseline-1.0-parquet", None, "train", "text", 0.25),
    ("cosmopedia", "HuggingFaceTB/smollm-corpus", "cosmopedia-v2", "train", "text", 0.10),
    ("gutenberg", "manu/project_gutenberg", None, "en", "text", 0.10),
]
BOOK_CHUNK_CHARS = 16_000  # ~4k tokens, so one novel cant fill a whole run of batches
EVAL_SEQS, EVAL_DOCS = 256, 1500  # held out from the start of fineweb-edu
WIKI_EVAL = ("Salesforce/wikitext", "wikitext-103-raw-v1", "test")  # in none of the sources, fair to compare against lilbase
SHUFFLE_BUFFER = 5000
RAM_FLOOR_GB = 2.0

TOKENIZER = "hf-internal-testing/llama-tokenizer"
OUT = "/kaggle/working/lilbase2"
RESUME_FROM = ""  # "" = newest checkpoint found anywhere, or a path like "/kaggle/input/.../lilbase2_step_00004000"
RESUME_GLOB = "/kaggle/input/**/lilbase2_step_*.json"
MAX_HOURS = 8.4  # sessions die at 9h. this leaves time to save
os.makedirs(OUT, exist_ok=True)

TOKENS_PER_STEP = BS * GRAD_ACC * SEQ * N_CHIPS
TOTAL_STEPS = TOTAL_TOKENS // TOKENS_PER_STEP
DECAY_STEPS = int(TOTAL_STEPS * DECAY_FRAC)
DECAY_START = TOTAL_STEPS - DECAY_STEPS
EVAL_BS = N_CHIPS * BS
DT = jnp.bfloat16 if DEV.platform == "tpu" else jnp.float32
MODEL_CONFIG = {"VOCAB": VOCAB, "D": D, "N_LAYERS": N_LAYERS, "N_HEADS": N_HEADS, "N_KV": N_KV, "D_FF": D_FF, "SEQ": SEQ}

assert D % N_HEADS == 0 and N_HEADS % N_KV == 0
assert WARMUP_STEPS < DECAY_START, "warmup runs into the decay, raise TOTAL_TOKENS or lower WARMUP_STEPS"
assert abs(sum(s[-1] for s in SOURCES) - 1) < 1e-6, "SOURCES shares should add up to 1"


class MemThread:
    def __init__(self):
        self.phase, self.low, self.peak = "setup", False, 0.0
        threading.Thread(target=self._run, daemon=True).start()

    # samples every second so compile and eval spikes get caught too, not just the steady state
    def _run(self):
        proc = psutil.Process()
        while True:
            rss = proc.memory_info().rss / 2**30
            free = psutil.virtual_memory().available / 2**30
            self.peak = max(self.peak, rss)
            low = free < RAM_FLOOR_GB
            if low and not self.low:
                print(f"[ram] only {free:.1f} GB free during '{self.phase}', this process holds {rss:.1f} GB", flush=True)
            self.low = low
            time.sleep(1)


if "MEM" not in globals():
    MEM = MemThread()

# %%
def rms_norm(x, w):
    xf = x.astype(jnp.float32)
    return xf * jax.lax.rsqrt(jnp.mean(xf * xf, -1, keepdims=True) + NORM_EPS) * w


# bf16 matmul, fp32 accumulate
def linear(x, w, out=None):
    return jnp.einsum("...i,oi->...o", x.astype(DT), w.astype(DT), preferred_element_type=out or DT)


def rope(x, cos, sin):
    half = HEAD_DIM // 2
    x1, x2 = x[..., :half].astype(jnp.float32), x[..., half:].astype(jnp.float32)
    return jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


def rope_tables(seq_len):
    inv = ROPE_THETA ** (-jnp.arange(0, HEAD_DIM, 2, dtype=jnp.float32) / HEAD_DIM)
    t = jnp.arange(seq_len, dtype=jnp.float32)[:, None] * inv[None]
    return jnp.cos(t), jnp.sin(t)


def block(p, x, cos, sin):
    B, L, _ = x.shape
    h = rms_norm(x, p["attn_norm.weight"])
    q = linear(h, p["attn.q_proj.weight"]).reshape(B, L, N_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    k = linear(h, p["attn.k_proj.weight"]).reshape(B, L, N_KV, HEAD_DIM).transpose(0, 2, 1, 3)
    v = linear(h, p["attn.v_proj.weight"]).reshape(B, L, N_KV, HEAD_DIM).transpose(0, 2, 1, 3)
    q, k = rope(q, cos, sin).astype(DT), rope(k, cos, sin).astype(DT)
    k = jnp.repeat(k, N_HEADS // N_KV, axis=1)
    v = jnp.repeat(v, N_HEADS // N_KV, axis=1)
    s = jnp.einsum("bhqd,bhkd->bhqk", q, k, preferred_element_type=jnp.float32) * HEAD_DIM ** -0.5
    s = jnp.where(jnp.tril(jnp.ones((L, L), dtype=bool)), s, -jnp.inf)
    a = jax.nn.softmax(s, axis=-1).astype(DT)
    o = jnp.einsum("bhqk,bhkd->bhqd", a, v).transpose(0, 2, 1, 3).reshape(B, L, -1)
    x = x + linear(o, p["attn.o_proj.weight"], jnp.float32)
    h = rms_norm(x, p["mlp_norm.weight"])
    m = jax.nn.silu(linear(h, p["mlp.gate_proj.weight"])) * linear(h, p["mlp.up_proj.weight"])
    return x + linear(m, p["mlp.down_proj.weight"], jnp.float32)


# scan traces block() once and reuses it for every layer, compile time went from forever to fine
def forward(params, ids, remat=True):
    cos, sin = rope_tables(ids.shape[1])
    x = params["embed_tokens.weight"][ids].astype(jnp.float32)
    layers = {k[len("layers."):]: v for k, v in params.items() if k.startswith("layers.")}
    body = lambda x, p: (block(p, x, cos, sin), None)
    if remat:
        body = jax.checkpoint(body, policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
    x, _ = jax.lax.scan(body, x, layers)
    x = rms_norm(x, params["norm.weight"])
    return linear(x, params["embed_tokens.weight"], jnp.float32)


def loss_fn(params, seqs, remat=True):
    logits = forward(params, seqs[:, :-1], remat)
    tgt = seqs[:, 1:]
    nll = jax.nn.logsumexp(logits, -1) - jnp.take_along_axis(logits, tgt[..., None], -1)[..., 0]
    return nll.mean()


def shapes():
    kv, L = N_KV * HEAD_DIM, N_LAYERS
    return {
        "embed_tokens.weight": (VOCAB, D), "norm.weight": (D,),
        "layers.attn_norm.weight": (L, D), "layers.mlp_norm.weight": (L, D),
        "layers.attn.q_proj.weight": (L, D, D), "layers.attn.k_proj.weight": (L, kv, D),
        "layers.attn.v_proj.weight": (L, kv, D), "layers.attn.o_proj.weight": (L, D, D),
        "layers.mlp.gate_proj.weight": (L, D_FF, D), "layers.mlp.up_proj.weight": (L, D_FF, D),
        "layers.mlp.down_proj.weight": (L, D, D_FF),
    }


# zero-1: adam state is split across chips along the first axis that divides evenly.
# layer stacks are (28, ...) so they split on axis 1
def shard_axis(shape):
    for ax, n in enumerate(shape):
        if n % N_CHIPS == 0:
            return ax
    raise ValueError(f"no axis of {shape} splits across {N_CHIPS} chips")


AXES = {k: shard_axis(s) for k, s in shapes().items()}
OPT_SPEC = {k: P(*[("data" if i == AXES[k] else None) for i in range(len(s))]) for k, s in shapes().items()}
OPT_SHARDING = {k: jax.sharding.NamedSharding(mesh, spec) for k, spec in OPT_SPEC.items()}


def decays(k):
    return not k.endswith("norm.weight")


# adamw on this chip's slice only, then all_gather the updated slices back into full weights
def adamw_update(params, g, opt, lr):
    idx = jax.lax.axis_index("data")
    step = opt["step"] + 1
    gnorm = jnp.sqrt(jax.lax.psum(sum(jnp.sum(x * x) for x in g.values()), "data"))
    scale = jnp.minimum(1.0, GRAD_CLIP / (gnorm + 1e-6))
    t = step.astype(jnp.float32)
    c1, c2 = 1 - BETA1 ** t, 1 - BETA2 ** t
    new_p, m, v = {}, {}, {}
    for k, p in params.items():
        ax, n = AXES[k], g[k].shape[AXES[k]]
        p_mine = jax.lax.dynamic_slice_in_dim(p, idx * n, n, axis=ax)
        gk = g[k] * scale
        m[k] = BETA1 * opt["m"][k] + (1 - BETA1) * gk
        v[k] = BETA2 * opt["v"][k] + (1 - BETA2) * gk * gk
        upd = (m[k] / c1) / (jnp.sqrt(v[k] / c2) + ADAM_EPS)
        if decays(k):
            upd = upd + WEIGHT_DECAY * p_mine
        new_p[k] = jax.lax.all_gather(p_mine - lr * upd, "data", axis=ax, tiled=True)
    return new_p, {"step": step, "m": m, "v": v}, gnorm


NO_REP_CHECK = {"check_vma" if "check_vma" in inspect.signature(jax.shard_map).parameters else "check_rep": False}


# shard_map so each chip scans its own accum steps. grads are reduce-scattered so each chip only gets the slice it updates
@partial(jax.jit, donate_argnums=(0, 1))
def train_step(params, opt, batch, lr):
    opt_specs = {"step": P(), "m": OPT_SPEC, "v": OPT_SPEC}

    # replication check off: jax cant prove the all_gather output is identical on every chip, but it is
    @partial(jax.shard_map, mesh=mesh, in_specs=(P(), opt_specs, P("data"), P()), out_specs=(P(), opt_specs, P(), P()),
             **NO_REP_CHECK)
    def _step(params, opt, batch, lr):
        b = batch[0]
        def micro(acc, mb):
            l, g = jax.value_and_grad(loss_fn)(params, mb)
            return jax.tree.map(jnp.add, acc, g), l
        g, losses = jax.lax.scan(micro, jax.tree.map(jnp.zeros_like, params), b)
        denom = b.shape[0] * N_CHIPS
        g = {k: jax.lax.psum_scatter(x, "data", scatter_dimension=AXES[k], tiled=True) / denom for k, x in g.items()}
        loss = jax.lax.pmean(losses.mean(), "data")
        new_params, new_opt, gnorm = adamw_update(params, g, opt, lr)
        return new_params, new_opt, loss, gnorm
    return _step(params, opt, batch, lr)


@jax.jit
def val_loss(params, seqs):
    return loss_fn(params, seqs, remat=False)


def get_lr(step):
    if step < WARMUP_STEPS:
        return LR * (step + 1) / WARMUP_STEPS
    if step < DECAY_START:
        return LR
    pr = min(1.0, (step - DECAY_START + 1) / max(1, DECAY_STEPS))
    return LR * (1 - (1 - MIN_LR_RATIO) * pr)


def init_model(seed=0):
    rng = np.random.default_rng(seed)
    out_std = 0.02 / math.sqrt(2 * N_LAYERS)
    params = {}
    for k, s in shapes().items():
        if k.endswith("norm.weight"):
            arr = np.ones(s, np.float32)
        else:
            std = out_std if k.endswith(("o_proj.weight", "down_proj.weight")) else 0.02
            arr = rng.standard_normal(s, dtype=np.float32) * std
        params[k] = jax.device_put(arr, repl)
    return params


def init_adam():
    zeros = lambda: {k: jax.device_put(np.zeros(s, np.float32), OPT_SHARDING[k]) for k, s in shapes().items()}
    return {"step": jax.device_put(jnp.array(0, jnp.int32), repl), "m": zeros(), "v": zeros()}


N_PARAMS = sum(math.prod(s) for s in shapes().values())
print(f"{N_PARAMS / 1e6:.1f}M params, {TOTAL_STEPS} steps ({TOTAL_STEPS * TOKENS_PER_STEP / 1e9:.2f}B tokens, "
      f"{TOKENS_PER_STEP:,}/step over {N_CHIPS} chips), lr flat until step {DECAY_START} then decays")

# %%
tok = AutoTokenizer.from_pretrained(TOKENIZER)
tok.model_max_length = 10**9
EOS = tok.eos_token_id
TEXT_FEATURES = Features({"text": Value("string"), "src": Value("int32")})
GUTENBERG_START = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.I | re.S)
GUTENBERG_END = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG", re.I)


# strips the license boilerplate, unwraps the hard 70-char line breaks, cuts on paragraph breaks
def book_chunks(text):
    text = text.replace("\r\n", "\n")
    m = GUTENBERG_START.search(text)
    text = text[m.end():] if m else text
    m = GUTENBERG_END.search(text)
    text = text[:m.start()] if m else text
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text.strip())
    chunks, i = [], 0
    while i < len(text):
        j = text.find("\n\n", i + BOOK_CHUNK_CHARS)
        if j == -1 or j - i > 2 * BOOK_CHUNK_CHARS:
            j = min(len(text), i + 2 * BOOK_CHUNK_CHARS) if j == -1 else i + BOOK_CHUNK_CHARS
        chunks.append(text[i:j].strip())
        i = j
    return [c for c in chunks if c]


def open_source(i, skip=0):
    name, path, cfg, split, col, _ = SOURCES[i]
    ds = load_dataset(path, cfg, split=split, streaming=True)
    if skip:
        ds = ds.skip(skip)
    ds = ds.select_columns([col])
    if name == "gutenberg":
        fn = lambda b: (lambda cs: {"text": cs, "src": [i] * len(cs)})([c for t in b[col] for c in book_chunks(t or "")])
        return ds.map(fn, batched=True, batch_size=1, remove_columns=[col] if col != "text" else None, features=TEXT_FEATURES)
    fn = lambda b: {"text": [t or "" for t in b[col]], "src": [i] * len(b[col])}
    return ds.map(fn, batched=True, remove_columns=[col] if col != "text" else None, features=TEXT_FEATURES)


# pulls a few rows from every source so a wrong name or column fails now, not 3 hours in.
# also measures tokens per example, since interleave samples examples, not tokens
def preflight(n=200):
    lens = {}
    for i, s in enumerate(SOURCES):
        try:
            rows = list(open_source(i).take(n))
        except Exception as e:
            raise RuntimeError(f"couldnt read {s[0]} ({s[1]}, config={s[2]}, split={s[3]}, column={s[4]}). "
                               f"fix or swap it in SOURCES: {type(e).__name__}: {e}") from e
        enc = tok([r["text"] for r in rows if r["text"]], add_special_tokens=False)["input_ids"]
        if len(enc) < n // 2:
            raise RuntimeError(f"{s[0]} only gave {len(enc)} non-empty rows out of {n}, check the text column")
        lens[s[0]] = float(np.mean([len(e) + 1 for e in enc]))
        print(f"  {s[0]:12s} ok, {lens[s[0]]:7.0f} tokens/example")
    w = [s[-1] / lens[s[0]] for s in SOURCES]
    return [x / sum(w) for x in w]


def train_stream(probs):
    parts = [open_source(i, skip=EVAL_DOCS if i == 0 else 0) for i in range(len(SOURCES))]
    ds = interleave_datasets(parts, probabilities=probs, seed=0, stopping_strategy="first_exhausted")
    return ds.shuffle(buffer_size=SHUFFLE_BUFFER, seed=0)


def to_seqs(texts, n_seqs):
    need, ids = n_seqs * (SEQ + 1), []
    for t in texts:
        ids.extend(tok(t, add_special_tokens=False)["input_ids"])
        ids.append(EOS)
        if len(ids) >= need:
            break
    n = min(n_seqs, len(ids) // (SEQ + 1)) // EVAL_BS * EVAL_BS
    if n == 0:
        raise RuntimeError(f"only {len(ids)} eval tokens, not enough for one batch")
    return np.array(ids[:n * (SEQ + 1)], np.int32).reshape(n, SEQ + 1)


def make_evals():
    fw = to_seqs((r["text"] for r in open_source(0).take(EVAL_DOCS)), EVAL_SEQS)
    wiki_rows = load_dataset(*WIKI_EVAL[:2], split=WIKI_EVAL[2], streaming=True)
    lines = [r["text"] for r in wiki_rows]
    wiki = to_seqs(("".join(lines[i:i + 2000]) for i in range(0, len(lines), 2000)), EVAL_SEQS)
    return {"fineweb": fw, "wikitext": wiki}


class Loader:
    def __init__(self, probs, state=None, counts=None):
        self.probs, self.state = probs, state
        self.counts = list(counts or [0] * len(SOURCES))
        self.q = queue.Queue(maxsize=32)
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _open(self):
        ds = train_stream(self.probs)
        if self.state is not None:
            ds.load_state_dict(self.state)
        return ds

    # tokenizes in the background. network errors reopen the stream at the last emitted batch
    def _run(self):
        need = N_CHIPS * GRAD_ACC * BS * (SEQ + 1)
        failures = 0
        while not self.stop.is_set():
            buf = np.empty(0, np.int32)
            try:
                ds = self._open()
                it = iter(ds)
                while not self.stop.is_set():
                    while buf.size < need:
                        rows = [next(it) for _ in range(64)]
                        enc = tok([r["text"] for r in rows], add_special_tokens=False)["input_ids"]
                        for r, e in zip(rows, enc):
                            self.counts[r["src"]] += len(e) + 1
                        buf = np.concatenate([buf] + [np.array(e + [EOS], np.int32) for e in enc])
                    self.state = ds.state_dict()
                    item = (buf[:need].reshape(N_CHIPS, GRAD_ACC, BS, SEQ + 1).copy(), self.state, list(self.counts))
                    buf = buf[need:]
                    failures = 0
                    while not self.stop.is_set():
                        try:
                            self.q.put(item, timeout=1)
                            break
                        except queue.Full:
                            continue
            except StopIteration:
                self.q.put(RuntimeError("a training source ran out of documents"))
                return
            except Exception as e:
                failures += 1
                if failures > 5:
                    self.q.put(e)
                    return
                print(f"[data] {type(e).__name__}: {e}; reopening stream in {30 * failures}s ({failures}/5)", flush=True)
                self.stop.wait(30 * failures)

    def get(self):
        item = self.q.get()
        if isinstance(item, BaseException):
            raise RuntimeError("data thread died") from item
        return item

    def close(self):
        self.stop.set()
        self.thread.join(timeout=10)


MEM.phase = "build eval sets"
if "EVALS" not in globals():
    EVALS = make_evals()
print("held-out: " + ", ".join(f"{k} {v.shape[0]} x {SEQ} tokens" for k, v in EVALS.items()))

# %%
def ckpt_path(step):
    return os.path.join(OUT, f"lilbase2_step_{step:08d}")


# zero padded so max() by filename gets the newest, wherever it lives
def find_ckpt():
    if RESUME_FROM:
        name = RESUME_FROM[:-5] if RESUME_FROM.endswith(".json") else RESUME_FROM.removesuffix(".safetensors")
        if not os.path.exists(name + ".json") or not os.path.exists(name + ".safetensors"):
            raise FileNotFoundError(f"RESUME_FROM={RESUME_FROM!r} but {name}.json/.safetensors arent both there")
        return name
    found = [f for f in glob.glob(os.path.join(OUT, "lilbase2_step_*.json")) + glob.glob(RESUME_GLOB, recursive=True)
             if os.path.exists(f[:-5] + ".safetensors")]
    return max(found, key=os.path.basename)[:-5] if found else None


# (L, ...) stacked -> layers.0.x, layers.1.x ...
def split_layers(d, prefix):
    out = {}
    for k, v in d.items():
        v = np.asarray(v)
        if k.startswith("layers."):
            for i in range(N_LAYERS):
                out[f"{prefix}layers.{i}.{k[len('layers.'):]}"] = v[i]
        else:
            out[prefix + k] = v
    return out


def stack_layers(d, place):
    out, layers_by_name = {}, {}
    for k, v in d.items():
        parts = k.split(".")
        if parts[0] == "layers":
            layers_by_name.setdefault(".".join(parts[2:]), {})[int(parts[1])] = v
        else:
            out[k] = v
    for k, layers in layers_by_name.items():
        if sorted(layers) != list(range(N_LAYERS)):
            raise KeyError(f"missing layers for {k}")
        out["layers." + k] = np.stack([layers[i] for i in range(N_LAYERS)])
    missing, extra = set(shapes()) - set(out), set(out) - set(shapes())
    if missing or extra:
        raise KeyError(f"checkpoint doesnt match the model: missing {sorted(missing)[:3]}, extra {sorted(extra)[:3]}")
    for k, s in shapes().items():
        if out[k].shape != s:
            raise ValueError(f"{k} is {out[k].shape} in the checkpoint but {s} in the config")
    return {k: jax.device_put(out[k], place(k)) for k in shapes()}


# write to tmp then rename, so a crash mid-save cant corrupt it. json last, so a json means the weights are complete
def save_st(name, flat, meta):
    save_file(flat, name + ".safetensors.tmp")
    os.replace(name + ".safetensors.tmp", name + ".safetensors")
    with open(name + ".json.tmp", "w") as f:
        json.dump(meta, f)
    os.replace(name + ".json.tmp", name + ".json")


# p: weights, m:/v: adam state. only the newest is kept in OUT
def save_ckpt(params, opt, step, run):
    flat = split_layers(params, "p:")
    flat.update(split_layers(opt["m"], "m:"))
    flat.update(split_layers(opt["v"], "v:"))
    name = ckpt_path(step)
    save_st(name, flat, {
        "step": step, "opt_step": int(opt["step"]), "total_steps": TOTAL_STEPS, "decay_start": DECAY_START,
        "tokens_seen": step * TOKENS_PER_STEP, "dataset_state": run["stream_pos"], "source_probs": run["probs"],
        "source_tokens": run["counts"], "sources": [list(s) for s in SOURCES], "history": run["history"],
        "datasets_version": datasets.__version__, "transformers_version": transformers.__version__,
        "model_config": MODEL_CONFIG,
    })
    del flat
    for old in glob.glob(os.path.join(OUT, "lilbase2_step_*")):
        if not old.startswith(name + "."):
            os.remove(old)
    run["saved_step"] = step
    print(f"[checkpoint] step {step} -> {name}.safetensors", flush=True)


def load_ckpt(name):
    with open(name + ".json") as f:
        meta = json.load(f)
    if meta["model_config"] != MODEL_CONFIG:
        raise ValueError(f"checkpoint is {meta['model_config']} but the config cell says {MODEL_CONFIG}")
    if [list(s) for s in SOURCES] != meta["sources"]:
        raise ValueError("SOURCES changed since this checkpoint, so its saved stream position wont line up. put them back")
    if meta["datasets_version"] != datasets.__version__:
        raise ValueError(f"checkpoint was made with datasets {meta['datasets_version']}, this is {datasets.__version__}")
    old_start = meta["decay_start"]
    if meta["total_steps"] != TOTAL_STEPS and meta["step"] > min(old_start, DECAY_START):
        raise ValueError(f"TOTAL_TOKENS changed after the lr decay began (was {meta['total_steps']} steps, now {TOTAL_STEPS}). "
                         "put it back, changing it now would jump the learning rate")
    if meta["total_steps"] != TOTAL_STEPS:
        print(f"schedule changed: {meta['total_steps']} -> {TOTAL_STEPS} steps, decay now starts at {DECAY_START}")
    flat = load_file(name + ".safetensors")
    params = stack_layers({k[2:]: v for k, v in flat.items() if k.startswith("p:")}, lambda k: repl)
    opt = {"step": jax.device_put(jnp.array(meta["opt_step"], jnp.int32), repl),
           "m": stack_layers({k[2:]: v for k, v in flat.items() if k.startswith("m:")}, OPT_SHARDING.get),
           "v": stack_layers({k[2:]: v for k, v in flat.items() if k.startswith("v:")}, OPT_SHARDING.get)}
    del flat
    return params, opt, meta


def ram_used():
    vm = psutil.virtual_memory()
    return (vm.total - vm.available) / 2**30, vm.available / 2**30


def hbm_used():
    ms = DEV.memory_stats() or {}
    return ms.get("peak_bytes_in_use", 0) / 2**30


# eval batches are split across chips, params stay replicated
def evaluate(params):
    out = {}
    for k, seqs in EVALS.items():
        losses = [val_loss(params, jax.device_put(seqs[i:i + EVAL_BS], sharded)) for i in range(0, len(seqs), EVAL_BS)]
        out[k] = float(np.mean([float(x) for x in losses]))
    return out


def mix_str(counts):
    tot = max(1, sum(counts))
    return "  ".join(f"{s[0]} {100 * c / tot:.1f}%" for s, c in zip(SOURCES, counts))

# %%
MEM.phase = "load/init weights"
name = find_ckpt()
if name:
    params, opt, meta = load_ckpt(name)
    step = meta["step"]
    run = {"stream_pos": meta["dataset_state"], "probs": meta["source_probs"], "counts": meta["source_tokens"],
           "history": meta["history"]}
    print(f"resuming from step {step}/{TOTAL_STEPS} ({100 * step / TOTAL_STEPS:.1f}%, "
          f"{step * TOKENS_PER_STEP / 1e9:.2f}B tokens seen) <- {name}")
    # copy it into this version's output right away, so even a session that dies early leaves something to resume from
    if os.path.dirname(os.path.abspath(name)) != os.path.abspath(OUT):
        for ext in (".safetensors", ".json"):
            shutil.copyfile(name + ext, ckpt_path(step) + ext)
        run["saved_step"] = step
else:
    print("starting from scratch. checking sources:")
    probs = preflight()
    params = init_model(seed=0)
    opt = init_adam()
    step = 0
    run = {"stream_pos": None, "probs": probs, "counts": [0] * len(SOURCES), "history": {"train": [], "eval": []}}
    print("sampling odds per example: " + "  ".join(f"{s[0]} {p:.3f}" for s, p in zip(SOURCES, probs)))

tot, cnt = jnp.float32(0), 0
timed_out = False
SPLIT = 1
if step < TOTAL_STEPS:
    loader = Loader(run["probs"], run["stream_pos"], run["counts"])
    print(f"training steps {step} -> {TOTAL_STEPS} (first step compiles, give it a minute or two)")
    MEM.phase = "compile + first step"
    try:
        while step < TOTAL_STEPS:
            if MEM.low:
                raise MemoryError("host RAM nearly full")
            if time.time() - RUN_START > MAX_HOURS * 3600:
                timed_out = True
                break

            batch, stream_pos, counts = loader.get()
            # if hbm overflows on compile, halve micro and double accum
            while True:
                try:
                    xb = jax.device_put(jnp.asarray(batch.reshape(N_CHIPS, GRAD_ACC * SPLIT, BS // SPLIT, SEQ + 1)), sharded)
                    params, opt, loss, gnorm = train_step(params, opt, xb, jnp.float32(get_lr(step)))
                    break
                except jax.errors.JaxRuntimeError as e:
                    if "RESOURCE_EXHAUSTED" not in str(e) or MEM.phase == "train" or (BS // SPLIT) % 2:
                        raise
                    if any(x.is_deleted() for x in jax.tree.leaves((params, opt))):
                        raise RuntimeError("hbm overflow after the weights were donated, lower BS and rerun") from e
                    SPLIT *= 2
                    print(f"hbm overflow while compiling, retrying with micro-batch {BS // SPLIT} x {GRAD_ACC * SPLIT} accum")
            step += 1
            run["stream_pos"], run["counts"] = stream_pos, counts
            if MEM.phase != "train":
                loss.block_until_ready()
                print(f"compiled. host ram peak {MEM.peak:.1f} GB, hbm peak {hbm_used():.1f} GB")
                MEM.phase, t0 = "train", time.time()
            else:
                tot, cnt = tot + loss, cnt + 1

            # q near 0 for long stretches means tokenization is the bottleneck, not the tpu
            if cnt and (step % PRINT_EVERY == 0 or step == TOTAL_STEPS):
                avg = float(tot) / cnt
                dt = (time.time() - t0) / cnt
                run["history"]["train"].append([step, avg])
                if step % STATS_EVERY == 0 or step == TOTAL_STEPS:
                    left_h = (TOTAL_STEPS - step) * dt / 3600
                    used, _ = ram_used()
                    print(f"step {step:6d}/{TOTAL_STEPS}  loss={avg:.4f}  gnorm={float(gnorm):.2f}  lr={get_lr(step - 1):.2e}  "
                          f"{dt:.3f}s/step  {TOKENS_PER_STEP / dt:,.0f} tok/s  left={left_h:.1f}h (~{math.ceil(left_h / MAX_HOURS)} sessions)  "
                          f"ram={used:.1f}GB  hbm_peak={hbm_used():.1f}GB  q={loader.q.qsize()}", flush=True)
                else:
                    print(f"step {step:6d}/{TOTAL_STEPS}  loss={avg:.4f}", flush=True)
                if not math.isfinite(avg):
                    raise FloatingPointError(f"loss is {avg} at step {step}")
                tot, cnt, t0 = jnp.float32(0), 0, time.time()

            if step % EVAL_EVERY == 0 or step == TOTAL_STEPS:
                MEM.phase = "eval"
                ev = evaluate(params)
                run["history"]["eval"].append([step, ev["fineweb"], ev["wikitext"]])
                print(f"[eval] step {step}  " + "  ".join(f"{k} loss={v:.4f} ppl={math.exp(v):.1f}" for k, v in ev.items()))
                print(f"[mix]  {mix_str(run['counts'])}", flush=True)
                MEM.phase, t0 = "train", time.time()

            if step % SAVE_EVERY == 0 or step == TOTAL_STEPS:
                save_ckpt(params, opt, step, run)
                t0 = time.time()
    # dont save on nan, it would overwrite the last good checkpoint
    except FloatingPointError:
        raise
    except BaseException as e:
        if any(x.is_deleted() for x in jax.tree.leaves((params, opt))):
            print(f"{type(e).__name__} mid-step, params gone. last checkpoint: {find_ckpt()}")
        elif run.get("saved_step") != step:
            save_ckpt(params, opt, step, run)
            print(f"{type(e).__name__} at step {step}; checkpoint saved")
        raise
    finally:
        loader.close()

if timed_out:
    if run.get("saved_step") != step:
        save_ckpt(params, opt, step, run)
    print(f"stopped at step {step}/{TOTAL_STEPS} to beat the 9h limit. to continue: add input > your work > "
          "this notebook (newest version), then save version again")
elif step >= TOTAL_STEPS:
    print(f"training done at step {step}")

# %%
import matplotlib.pyplot as plt

tr, ev = np.array(run["history"]["train"]), np.array(run["history"]["eval"])
plt.figure(figsize=(8, 4))
if len(tr):
    plt.plot(tr[:, 0], tr[:, 1], label=f"train ({PRINT_EVERY}-step mean)", alpha=0.6)
if len(ev):
    plt.plot(ev[:, 0], ev[:, 1], "o-", label="fineweb-edu held-out")
    plt.plot(ev[:, 0], ev[:, 2], "s-", label="wikitext-103")
if len(tr) or len(ev):
    plt.axvline(DECAY_START, color="gray", ls=":", label="decay starts")
plt.xlabel("step"); plt.ylabel("loss"); plt.ylim(top=min(8, plt.ylim()[1])); plt.legend(); plt.grid(alpha=0.3)
plt.savefig(os.path.join(OUT, "loss.png"), dpi=120, bbox_inches="tight")
plt.show()

# %%
SAMPLE_LEN = 256


@jax.jit
def next_logits(params, ids, pos):
    return forward(params, ids, remat=False)[0, pos]


# fixed length so it only compiles once. starts with eos since thats what training docs start after
def sample(prompt, n=80, temp=0.8, top_k=40, seed=0):
    rng = np.random.default_rng(seed)
    ids = [EOS] + tok(prompt, add_special_tokens=False)["input_ids"][-(SAMPLE_LEN - n - 1):]
    buf = np.zeros((1, SAMPLE_LEN), np.int32)
    buf[0, :len(ids)] = ids
    end = len(ids)
    while end < min(len(ids) + n, SAMPLE_LEN):
        lg = np.asarray(next_logits(params, jnp.asarray(buf), end - 1)) / temp
        top = np.argpartition(lg, -top_k)[-top_k:]
        p = np.exp(lg[top] - lg[top].max())
        nxt = int(rng.choice(top, p=p / p.sum()))
        if nxt == EOS:
            break
        buf[0, end] = nxt
        end += 1
    return tok.decode(buf[0, 1:end])


for prompt in ["The water cycle begins when", "In 1905, Albert Einstein", "She opened the letter and"]:
    print(sample(prompt), "\n---")

# %%
# weights only (~2.1 gb), once training is finished
if step >= TOTAL_STEPS:
    name = find_ckpt()
    with open(name + ".json") as f:
        meta = json.load(f)
    export = os.path.join(OUT, f"lilbase2_weights_step_{meta['step']:08d}")
    weights = {k: v for k, v in load_file(name + ".safetensors").items() if k.startswith("p:")}
    print(f"exporting {len(weights)} tensors, dtype {next(iter(weights.values())).dtype}")
    save_st(export, weights, {**{k: meta[k] for k in ("step", "opt_step", "tokens_seen", "source_tokens", "sources")},
                              "model_config": meta["model_config"]})
    del weights
else:
    print(f"not exporting yet, {TOTAL_STEPS - step} steps to go")
