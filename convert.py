import argparse
import json
from pathlib import Path

from safetensors.numpy import load_file, save_file
from transformers import AutoTokenizer, LlamaConfig

TOKENIZER = "hf-internal-testing/llama-tokenizer"
EOS = 2  # llama tokenizer's </s>; every training document starts right after it, so it doubles as bos

# training-code names -> transformers LlamaForCausalLM names; norms first so "attn." can't hit "attn_norm."
RENAMES = [("attn_norm.", "input_layernorm."), ("mlp_norm.", "post_attention_layernorm."), ("attn.", "self_attn.")]
LAYER_KEYS = ["input_layernorm", "post_attention_layernorm", "self_attn.q_proj", "self_attn.k_proj",
              "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


def hf_name(key):
    key = key.removeprefix("p:")
    for old, new in RENAMES:
        key = key.replace(old, new)
    return "model." + key


def newest_weights():
    found = sorted((Path(__file__).parent / "weights").glob("*.safetensors"))
    if not found:
        raise SystemExit("no .safetensors in weights/; pass --weights explicitly")
    return found[-1]  # zero-padded step numbers sort correctly


def convert(weights_path, out_dir, with_tokenizer=True):
    weights_path, out_dir = Path(weights_path), Path(out_dir)
    cfg = json.loads((weights_path.parent / "model_config.json").read_text())
    raw = load_file(weights_path)
    tensors = {hf_name(k): v for k, v in raw.items() if k.startswith("p:")}
    if not tensors:
        tensors = raw

    expected = {"model.embed_tokens.weight", "model.norm.weight"} | {
        f"model.layers.{i}.{k}.weight" for i in range(cfg["num_hidden_layers"]) for k in LAYER_KEYS}
    if set(tensors) != expected:
        raise KeyError(f"unexpected tensors: {sorted(set(tensors) ^ expected)[:5]}")

    config = LlamaConfig(
        vocab_size=tensors["model.embed_tokens.weight"].shape[0],
        hidden_size=cfg["hidden_size"], intermediate_size=cfg["intermediate_size"], num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"], num_key_value_heads=cfg["num_key_value_heads"],
        max_position_embeddings=cfg["max_position_embeddings"], rms_norm_eps=cfg["rms_norm_eps"], rope_theta=cfg["rope_theta"], hidden_act=cfg["hidden_act"],
        tie_word_embeddings=cfg["tie_word_embeddings"], bos_token_id=cfg["bos_token_id"], eos_token_id=cfg["eos_token_id"], architectures=cfg["architectures"],
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(out_dir)
    # no lm_head.weight: tie_word_embeddings makes transformers reuse embed_tokens for it on load
    save_file(tensors, str(out_dir / "model.safetensors"), metadata={"format": "pt"})
    if with_tokenizer:
        AutoTokenizer.from_pretrained(TOKENIZER).save_pretrained(out_dir)
    return config


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="convert a TPU weights export into a transformers/PyTorch Llama folder")
    ap.add_argument("--weights", help="a *_weights_step_*.safetensors with its .json beside it; default: newest in weights/")
    ap.add_argument("--out", default=str(Path(__file__).parent / "hf"))
    ap.add_argument("--no-tokenizer", action="store_true", help="skip downloading and saving the tokenizer")
    args = ap.parse_args()

    src = args.weights or newest_weights()
    config = convert(src, args.out, with_tokenizer=not args.no_tokenizer)
    print(f"{src} -> {args.out} ({config.num_hidden_layers} layers, d={config.hidden_size})")
