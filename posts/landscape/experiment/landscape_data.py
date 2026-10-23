"""Turn the 51×51 slices into the page's data: every sliced trace, its reasoning, and one frame per point of it.

    python landscape_data.py runs/hires2 runs/anim ... landscape/data.json   # run directories, each with selection.json and slices

Only slices in the plane of the pictures are used (directions 300 and 301); a trace needs three to animate.
"""
import json
import math
import sys
from pathlib import Path


def odds_grid(sl, lo=-25.0, hi=21.0):
    """Log-odds of the right answer over a slice, clipped so the broken edge does not flatten the range that matters."""
    if "log_odds" in sl:
        return [[round(min(hi, max(lo, v)), 4) for v in row] for row in sl["log_odds"]]
    odds = lambda l: -l - math.log(max(1e-300, -math.expm1(-l)))   # from −log p
    return [[round(min(hi, max(lo, odds(l))), 4) for l in row] for row in sl["loss"]]


def alive_of(sl):
    """Whether each copy still answers at all: log p that its reply begins a number."""
    if sl.get("numeric"):
        return [[round(v, 4) for v in row] for row in sl["numeric"]]
    return [[0.0 if l < 35 else -6.0 for l in row] for row in sl["loss"]]   # a stand-in from the loss


def number_of(reply):
    """The number a copy's greedy reply gives: the text before its first "}", spaces and commas removed, if that is all
    digits. None when the reply never closes the box or is not a number."""
    if "}" not in reply:
        return None
    s = reply.split("}")[0].replace(" ", "").replace(",", "")
    return s if s.isdigit() else None


def numbers_of(sl):
    """The number each recorded copy gives, as {(i, j): number or None} over the slice's grid indices, or None when the
    slice did not record replies."""
    r = sl.get("replies")
    if not r or not r.get("text"):
        return None
    idx = r["index"]
    return {(idx[a], idx[b]): number_of(s) for a, row in enumerate(r["text"]) for b, s in enumerate(row)}


def add_who(frames, answer, n=51):
    """Colour every copy by what it says: 0 the answer followed, 1..3 the three commonest other numbers over the whole
    trace (so a colour means one number in every frame), 4 another number, 5 no number. Where only a subgrid of replies
    was recorded, each cell takes the nearest recorded one. Encoded as a string of n×n digits per frame."""
    nums = [numbers_of(f.pop("_slice")) for f in frames]
    if any(m is None for m in nums):
        return
    target = answer.replace(" ", "").replace(",", "")
    count = {}
    for m in nums:
        for v in m.values():
            if v is not None and v != target:
                count[v] = count.get(v, 0) + 1
    top = [v for v, _ in sorted(count.items(), key=lambda kv: (-kv[1], kv[0]))[:3]]
    for f, m in zip(frames, nums):
        recorded = sorted({i for i, _ in m})
        nearest = [min(recorded, key=lambda r: (abs(r - g), r)) for g in range(n)]
        def cat(i, j):
            v = m[(nearest[i], nearest[j])]
            return 5 if v is None else 0 if v == target else top.index(v) + 1 if v in top else 4
        f["who"] = "".join(str(cat(i, j)) for i in range(n) for j in range(n))
        f["who_numbers"] = top


def locate(think, tail):
    """Character index in the thinking where a prefix ending in `tail` stops."""
    if not tail:
        return 0
    for n in (200, 120, 60, 30):
        i = think.find(tail[-n:].strip())
        if i >= 0:
            return i + len(tail[-n:].strip())
    return None


def frame_of(sl, think, f, end=None):
    """One frame of the page: where the reasoning has got to, and the slice's two grids."""
    if end is None:
        end = 0 if f == 0 else locate(think, sl["prefix_tail"])
    return {"fraction": f, "end": end if end is not None else round(len(think) * f), "surface": odds_grid(sl), "alive": alive_of(sl), "_slice": sl}


def animations(src, sel):
    """Traces built from several selection entries sharing an "anim" key: each entry is the reasoning cut at one point, with
    the slice made for the answer the model gives there (right or wrong), so the frames follow that answer."""
    groups = {}
    for ti, tr in enumerate(sel):
        if "anim" in tr:
            groups.setdefault(tr["anim"], []).append((ti, tr))
    out = []
    for key, members in groups.items():
        full = max((tr["full_think"] for _, tr in members), key=len)
        frames = []
        for ti, tr in sorted(members, key=lambda m: len(m[1]["think"])):
            files = list(src.glob(f"surface-t{ti}-f1.000-g51-*.json"))
            if files:
                fr = frame_of(json.loads(files[0].read_text()), full, len(tr["think"]) / len(full), end=len(tr["think"]))
                fr["page"] = tr["think"]   # the page this frame was made from, for figures that show it
                frames.append(fr)
        tr = members[0][1]
        add_who(frames, tr["a"])
        if len(frames) == 1:   # a single page: shown as a still, which the viewer needs two frames for
            frames = [{**frames[0], "fraction": 0.0}, {**frames[0], "fraction": 1.0}]
        if frames:
            out.append({"key": key, "q": tr["q"], "a": tr["a"], "right": tr["right"], "kind": tr["kind"], "think": full, "frames": frames,
                        "legend_right": None if tr["a"] == tr["right"] else f"the answer it gives, {tr['a']} (wrong)"})
    return out


*srcs, dst = [Path(a) for a in sys.argv[1:]]
traces = []
for src in srcs:
    if not (src / "selection.json").exists():
        continue
    sel = json.loads((src / "selection.json").read_text())["traces"]
    traces += animations(src, sel)
    for ti, tr in enumerate(sel):
        if "anim" in tr:
            continue
        fine = {}
        for p in src.glob(f"surface-t{ti}-f*-g51-*.json"):
            if "-p" not in p.name:
                s = json.loads(p.read_text()); fine[round(s["fraction"], 3)] = s
        if len(fine) < 3:
            continue
        frames = [frame_of(sl, tr["think"], f) for f, sl in sorted(fine.items())]
        add_who(frames, tr["a"])
        traces.append({"q": tr["q"], "a": tr["a"], "kind": "resolving" if tr["kind"] == "correct" else tr["kind"], "think": tr["think"], "frames": frames})
dst.write_text(json.dumps({"traces": traces}))
for t in traces:
    n = len(t["frames"][0]["surface"]) // 2
    print(f"{t['kind'][:24]:24s} {len(t['frames']):2d} frames, log-odds at the model {t['frames'][0]['surface'][n][n]:6.1f} → {t['frames'][-1]['surface'][n][n]:6.1f}: {t['q'][:70]}")
