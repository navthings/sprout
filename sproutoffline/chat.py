from mlx_lm import load, stream_generate
import os

model, tokenizer = load(".")
messages = []

print("Sprout")
print("/clear  clear conversation")
print("/help   show commands")
print("/bye    exit")
print()

while True:
    try:
        user = input("> ")
    except (EOFError, KeyboardInterrupt):
        print()
        break

    command = user.strip()

    if command == "/bye":
        break

    if command == "/clear":
        messages.clear()
        os.system("clear")
        continue

    if command == "/help":
        print("/clear  clear conversation")
        print("/help   show commands")
        print("/bye    exit")
        print()
        continue

    if not command:
        continue

    messages.append({"role": "user", "content": user})

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    response = ""

    for result in stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=256,
    ):
        print(result.text, end="", flush=True)
        response += result.text

    print()
    messages.append({"role": "assistant", "content": response})
