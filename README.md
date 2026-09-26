# sprout

the next one after [lilbase](https://github.com/navthings/lilbase). bigger base model trained on a kaggle tpu, plus the scripts that turn a base model into a chat model on a mac.

## tpu/ (base model, in training)

523m param llama-style model, 12b tokens. same recipe as lilbase but about 1.8x the size and 2x the data, with a mixed dataset instead of just fineweb-edu:

| source | share |
|---|---|
| fineweb-edu (sample-100BT) | 55% |
| dclm-baseline | 25% |
| cosmopedia-v2 | 10% |
| project gutenberg (english) | 10% |

28 layers, d=1280, 20 query heads, 4 kv heads, 1024 context, llama tokenizer. adamw with the optimizer state sharded over the 8 chips, wsd lr schedule (warmup, flat, decay over the last 15%).


## sft/ (chat finetune)

full sft in mlx on a macbook air (16gb), on smol-smoltalk, loss only on the assistant turns. llama-2 `[INST]` chat format.

```
cd sft
uv venv && uv pip install -r requirements.txt
python sft.py --hours 12       # works out the step count from measured speed
python chat.py                 # talk to it in the terminal
```

`--base` defaults to lilbase's hf folder (`../lilbase/mac/hf`), point it at any llama-style hf folder with the same tokenizer.

the first thing out of it is lilchat, which is lilbase finetuned:

```
ollama run navthings/lilchat
```

it gets the format right and says hi properly. facts and maths are mostly made up, it's 297m params. same script goes on the new base model once it finishes.


## ambitions

the goals for sprout are to:

1. beat gpt2 extralarge (1.5b) or get close to it
2. finetune it to a competitive coding model for 500M
