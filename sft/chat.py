import argparse
from pathlib import Path

import mlx.core as mx

from sft import encode_chat, generate, load_model, load_tokenizer

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="chat with the sft'd lilbase in the terminal")
    ap.add_argument("--model", default=str(Path(__file__).parent / "hf-chat"))
    ap.add_argument("--system", help="optional system prompt")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--rep-penalty", type=float, default=1.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model, tok = load_model(args.model), load_tokenizer(args.model)
    ctx = model.config["max_position_embeddings"]
    mx.random.seed(args.seed)
    history = [{"role": "system", "content": args.system}] if args.system else []
    print("ctrl-d to quit, /reset to clear the conversation\n")

    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text == "/reset":
            history = history[:1] if args.system else []
            continue
        if not text:
            continue
        history.append({"role": "user", "content": text})
        ids = encode_chat(tok, history)
        # drop the oldest exchanges until there's room to answer
        while len(ids) > ctx - 128 and len(history) > (3 if args.system else 1):
            del history[int(bool(args.system)):int(bool(args.system)) + 2]
            ids = encode_chat(tok, history)

        out, shown = [], ""
        print("lilbase> ", end="", flush=True)
        for t in generate(model, ids, args.max_tokens, args.temp, args.top_k, args.rep_penalty):
            out.append(t)
            full = tok.decode(out)  # whole reply each time: sentencepiece spacing and multi-byte chars come out right
            if full.endswith("\ufffd"):
                continue
            print(full[len(shown):], end="", flush=True)
            shown = full
        print("\n")
        history.append({"role": "assistant", "content": shown})
