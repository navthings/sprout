# lilchat

[lilbase](https://ollama.com/navthings/lilbase) (297m params, trained from scratch) finetuned into a chat model on smol-smoltalk.

it follows the chat format, answers in full sentences, stops when it's done, and does lists and code blocks. it does not know things. expect confident wrong answers on facts and maths, and weak memory across turns.

```
ollama run navthings/lilchat
```

## tags

| tag | size | notes |
|---|---|---|
| `latest` / `q8_0` | 379mb | same quality as f16 |
| `q4_k_m` | 273mb | smallest |
| `f16` | 594mb | unquantized |

## finetune

full sft in mlx on a macbook air m5, 1,136 steps of 32k tokens (~37m tokens, about 12% of smol-smoltalk), loss only on assistant turns. llama-2 `[INST]` format, system prompts via `<<SYS>>`. held-out loss 1.337 vs 2.230 for lilbase on the same data.

## defaults

temperature 0.4, top-k 40, repeat penalty 1.1, 400 tokens, 1024 context. greedy decoding loops, keep some temperature.

## code

https://github.com/navthings/lilbase (`mac` branch: mac/sft.py, mac/chat.py)
