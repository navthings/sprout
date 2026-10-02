import json
import os

import chart

PW, PH, GAP, TOP, FOOT = 300, 230, 24, 46, 46
DARK = {"#0f0f0f": "#e6edf3", "#ffffff": "#0d1117", "#ececea": "#30363d", "#929292": "#8b949e",
        "#767676": "#8b949e", "#c9c9c6": "#6e7681"}


def build(res, dark=False):
    tasks = [t for t in chart.TASKS if t[0] != "hellaswag_acc_norm" or all(r.get("hella_v2") for r in res.values())]
    n = len(tasks)
    W = n * PW + (n - 1) * GAP + 24
    H = TOP + PH + FOOT
    font = "-apple-system, Segoe UI, Helvetica, Arial, sans-serif"
    s = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="{font}">'
    s += f'<rect width="{W}" height="{H}" fill="#ffffff"/>'
    for i, (key, title, sub) in enumerate(tasks):
        x0 = 12 + i * (PW + GAP)
        s += f'<text x="{x0 + 34}" y="20" font-size="14" font-weight="600" fill="#0f0f0f">{title}</text>'
        s += f'<text x="{x0 + 34}" y="37" font-size="12" fill="#767676">{sub}</text>'
        inner = chart.panel(res, key, title, sub)
        inner = inner[inner.index("<svg"):inner.index("</svg>") + 6]
        inner = inner.replace("<svg ", f'<svg x="{x0}" y="{TOP}" width="{PW}" height="{PH}" ', 1)
        s += inner
    n_gpt2 = sum(m in res for m in chart.GPT2)
    y = H - 16
    s += f'<circle cx="46" cy="{y - 4}" r="5" fill="#0f0f0f"/><text x="58" y="{y}" font-size="12" fill="#767676">sprout (chat version), 523m</text>'
    s += f'<circle cx="246" cy="{y - 4}" r="5" fill="#ffffff" stroke="#0f0f0f" stroke-width="1.6"/><text x="258" y="{y}" font-size="12" fill="#767676">lilbase, 297m</text>'
    s += f'<circle cx="366" cy="{y - 4}" r="5" fill="#c9c9c6"/><text x="378" y="{y}" font-size="12" fill="#767676">openai gpt-2, {n_gpt2} sizes</text>'
    s += "</svg>"
    if dark:
        for a, b in DARK.items():
            s = s.replace(a, b)
    return s


if __name__ == "__main__":
    res = json.load(open("results.json"))
    os.makedirs("../assets", exist_ok=True)
    open("../assets/vs_gpt2_light.svg", "w").write(build(res))
    open("../assets/vs_gpt2_dark.svg", "w").write(build(res, dark=True))
    print("wrote assets/vs_gpt2_light.svg and vs_gpt2_dark.svg")
