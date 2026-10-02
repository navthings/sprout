import json
import math
import sys

TASKS = [
    ("hellaswag_acc_norm", "hellaswag", "picking the sensible ending to a short scene"),
    ("arc_easy_acc_norm", "arc-easy", "grade school science questions"),
    ("lambada_acc", "lambada", "guessing the last word of a passage"),
]
GPT2 = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
GPT2_LABEL = {"gpt2": "small", "gpt2-medium": "medium", "gpt2-large": "large", "gpt2-xl": "xl"}
MINE = {"sprout-chat": "sprout", "lilbase": "lilbase"}

W, H = 300, 230
PAD = dict(l=34, r=14, t=16, b=40)
X_LO, X_HI = math.log10(100), math.log10(2000)


def nice_range(vals):
    lo = math.floor((min(vals) - 3) / 5) * 5
    hi = math.ceil((max(vals) + 3) / 5) * 5
    return lo, hi


def panel(res, key, title, sub):
    pts = {m: (res[m]["params_m"], res[m][key] * 100) for m in res if key in res[m]}
    lo, hi = nice_range([v for _, v in pts.values()])
    x = lambda p: PAD["l"] + (W - PAD["l"] - PAD["r"]) * (math.log10(p) - X_LO) / (X_HI - X_LO)
    y = lambda v: PAD["t"] + (H - PAD["t"] - PAD["b"]) * (1 - (v - lo) / (hi - lo))

    s = f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{title}: score against model size" font-family="Geist, Helvetica Neue, Helvetica, sans-serif" font-size="11" fill="#929292">'
    step = 5 if hi - lo <= 25 else 10
    t = lo
    while t <= hi:
        s += f'<line class="b-grid" stroke="#ececea" stroke-width="1" x1="{PAD["l"]}" x2="{W - PAD["r"]}" y1="{y(t):.1f}" y2="{y(t):.1f}"/>'
        s += f'<text x="{PAD["l"] - 6}" y="{y(t) + 4:.1f}" text-anchor="end">{t}%</text>'
        t += step
    for p, lab in [(124, "124m"), (355, "355m"), (774, "774m"), (1558, "1.5b")]:
        s += f'<text x="{x(p):.1f}" y="{H - 22}" text-anchor="middle">{lab}</text>'
    s += f'<text x="{(PAD["l"] + W - PAD["r"]) / 2:.1f}" y="{H - 4}" text-anchor="middle" class="b-axis" fill="#767676">model size (params)</text>'

    g = [m for m in GPT2 if m in pts]
    if len(g) > 1:
        d = " ".join(f'{"M" if i == 0 else "L"}{x(pts[m][0]):.1f} {y(pts[m][1]):.1f}' for i, m in enumerate(g))
        s += f'<path class="b-gpt2" fill="none" stroke="#c9c9c6" stroke-width="2" stroke-linejoin="round" d="{d}"/>'
    for m in g:
        px, py = x(pts[m][0]), y(pts[m][1])
        s += f'<circle class="b-gpt2-dot" fill="#c9c9c6" cx="{px:.1f}" cy="{py:.1f}" r="3.5"/>'
        s += f'<text class="b-gpt2-lab" font-size="10" x="{px:.1f}" y="{py + 15:.1f}" text-anchor="middle">{GPT2_LABEL[m]}</text>'
    for m, lab in MINE.items():
        if m not in pts:
            continue
        px, py = x(pts[m][0]), y(pts[m][1])
        cls = 'b-me" fill="#0f0f0f' if m == "sprout-chat" else 'b-me2" fill="#ffffff" stroke="#0f0f0f" stroke-width="1.6'
        s += f'<circle class="{cls}" cx="{px:.1f}" cy="{py:.1f}" r="5"/>'
        # label above the dot unless a gpt-2 dot or label sits there, then below
        taken = [(x(pts[g_][0]), y(pts[g_][1])) for g_ in g] + [(x(pts[g_][0]), y(pts[g_][1]) + 12) for g_ in g]
        clear = lambda ly: min((abs(ly - ty) + abs(px - tx) / 4 for tx, ty in taken), default=99)
        ly = max([py - 11, py + 19], key=clear)
        s += f'<text class="b-me-lab" fill="#0f0f0f" font-size="11.5" font-weight="500" x="{px:.1f}" y="{ly:.1f}" text-anchor="middle">{lab} {pts[m][1]:.1f}</text>'
    s += "</svg>"
    return f'<div class="bench-panel"><p class="bench-title"><b>{title}</b> <span>{sub}</span></p>{s}</div>'


if __name__ == "__main__":
    res = json.load(open(sys.argv[1]))
    panels = "".join(panel(res, *t) for t in TASKS)
    print(
        '<div class="fig bench" data-live>'
        f'<div class="bench-grid">{panels}</div>'
        '<div class="bench-key"><span><i class="k-me"></i>sprout (chat version), 523m</span>'
        '<span><i class="k-me2"></i>lilbase, 297m</span>'
        '<span><i class="k-gpt2"></i>openai gpt-2, 4 sizes</span></div>'
        '<p class="cap muted">higher is better. a dot above the grey line beats a gpt-2 of the same size. '
        "same test script for every model, on my macbook</p>"
        "</div>"
    )
