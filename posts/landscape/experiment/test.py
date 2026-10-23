"""The test: at matched confidence, does the reasoning change how robustly the answer is computed?

    python test.py runs/hires2 runs/many-2 ...     # every dense-t*.json under these

Per sentence boundary: confidence = log-odds of the right answer at the model; robustness = the median radius, over fixed
rays in weight space, at which the right answer stops being more likely than not (a ray that never flips within the
ladder counts as its end, so robustness is capped there). Then:
  - robustness at matched confidence, before and after the answer is first written in the reasoning
  - steps across single sentences where robustness moves without a comparable move in confidence, and whether they recur
  - the bending of the log-odds along the rays at every scale of the ladder, to see whether it survives as the step shrinks
Uncertainty comes from resampling questions, keeping each question's traces together.
"""
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

TAGS = [
    ("doubt", r"^\W*(wait|but wait|hold on|but hold on|hmm|hm+|oh|oops|no,|actually)\b"),
    ("verify", r"\b(let me (check|verify|double[- ]check|confirm|recheck)|double[- ]check|to verify|sanity check)\b"),
    ("alternative", r"^\W*(alternatively|another (way|approach)|or,? maybe|let me try (another|a different))\b"),
    ("conclude", r"^\W*(so|therefore|thus|hence|that means|this means)\b"),
    ("answer", r"(the answer is|final answer|\\boxed|answer should be)"),
]
BINS = [0, 2, 4, 7, 10, 15, math.inf]   # confidence bands, in nats of log-odds


def written_re(x):
    """The number as it might appear in the text, allowing thousands separators: 3432 also as 3,432 or 3 432."""
    d = x.lstrip("-")
    body = "".join(c + ("[, ]?" if i < len(d) - 1 and (len(d) - 1 - i) % 3 == 0 else "") for i, c in enumerate(d))
    return re.compile(rf"(?<![\d.,]){re.escape(x[:len(x) - len(d)])}{body}(?!\d|\.\d|,\d)")


def tag(sentence):
    s = sentence.strip().lower()
    for name, pat in TAGS:
        if re.search(pat, s):
            return name
    return "other"


def rows_of(files):
    rows = []
    for p in files:
        d = json.loads(p.read_text())
        radii, think, pts = d["radii"], d["think"], d["points"]
        for i, q in enumerate(pts):
            flips = [r["flip"] if r["flip"] is not None else radii[-1] for r in q["rays"]]
            rows.append({"q": d["q"], "trace": str(p), "i": i, "pos": q["end"] / len(think), "conf": q["log_odds"], "rob": float(np.median(flips)),
                         "capped": sum(r["flip"] is None for r in q["rays"]) / len(q["rays"]),
                         "written": written_re(d["a"]).search(think[:q["end"]]) is not None,   # from the text: the stored flag missed separators and full stops
                         "kind": d["kind"], "reply": q["reply_kind"], "tag": tag(think[pts[i - 1]["end"] if i else 0:q["end"]]),
                         "text": think[pts[i - 1]["end"] if i else 0:q["end"]].strip().replace("\n", " "), "rays": q["rays"], "radii": radii})
    return rows


def band(c):
    for lo, hi in zip(BINS, BINS[1:]):
        if lo <= c < hi:
            return f"[{lo}, {hi})" if hi < math.inf else f"{lo}+"
    return "below 0"


def expected(rows):
    """Robustness expected from confidence alone: the median within each confidence band, over every sentence."""
    by = {}
    for r in rows:
        by.setdefault(band(r["conf"]), []).append(r["rob"])
    return {b: float(np.median(v)) for b, v in by.items()}


def matched(rows, key, label):
    """Median robustness per confidence band, split by `key`."""
    print(f"\nrobustness at matched confidence, by {label} (median flip radius, n):")
    groups = sorted({r[key] for r in rows}, key=str)
    print(f"{'confidence':12s}" + "".join(f"{str(g):>22s}" for g in groups))
    for lo, hi in zip(BINS, BINS[1:]):
        cells = []
        for g in groups:
            v = [r["rob"] for r in rows if r[key] == g and lo <= r["conf"] < hi]
            cells.append(f"{np.median(v):8.3f} ({len(v):4d})" if v else f"{'—':>15s}")
        print(f"{band(lo):12s}" + "".join(f"{c:>22s}" for c in cells))


def steps(rows):
    """Changes across single sentences, within a trace."""
    out = []
    by = {}
    for r in rows:
        by.setdefault(r["trace"], []).append(r)
    for rs in by.values():
        rs.sort(key=lambda r: r["i"])
        for a, b in zip(rs, rs[1:]):
            out.append({**b, "dconf": b["conf"] - a["conf"], "drob": b["rob"] - a["rob"], "before": not b["written"]})   # the answer is still unwritten after the step
    return out


def events(st, drob=0.1, dconf=1.0):
    """Sentences across which robustness moves at least one ladder step while confidence moves less than `dconf` nats."""
    return [s for s in st if abs(s["drob"]) >= drob and abs(s["dconf"]) <= dconf]


def bootstrap(rows, stat, n=400, seed=0):
    rng = np.random.default_rng(seed)
    qs = sorted({r["q"] for r in rows})
    by = {q: [r for r in rows if r["q"] == q] for q in qs}
    vals = []
    for _ in range(n):
        pick = rng.choice(qs, len(qs), replace=True)
        v = stat([r for q in pick for r in by[q]])
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            vals.append(v)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (math.nan, math.nan)


def gap_before_after(rows):
    """Robustness before minus after the answer is written, at matched confidence, averaged over bands weighted by the smaller side."""
    num = den = 0
    for lo, hi in zip(BINS, BINS[1:]):
        a = [r["rob"] for r in rows if not r["written"] and lo <= r["conf"] < hi]
        b = [r["rob"] for r in rows if r["written"] and lo <= r["conf"] < hi]
        if len(a) >= 5 and len(b) >= 5:
            w = min(len(a), len(b)); num += w * (np.median(a) - np.median(b)); den += w
    return num / den if den else None


def curvature_by_scale(rows):
    """Second difference of the log-odds along each ray at each radius of the ladder, where both signs reached it."""
    print("\nbending of the log-odds along the rays, by scale (median over rays and sentences with confidence > 0; n):")
    radii = rows[0]["radii"]
    for r in radii:
        vals = []
        for row in rows:
            if row["conf"] <= 0:
                continue
            by_seed = {}
            for ray in row["rays"]:
                for x in ray["ladder"]:
                    if x["r"] == r:
                        by_seed.setdefault(ray["seed"], {})[ray["sign"]] = x["log_odds"]
            for v in by_seed.values():
                if "+" in v and "-" in v:
                    vals.append((v["+"] + v["-"] - 2 * row["conf"]) / r ** 2)
        if vals:
            print(f"  r = {r:<5}: {np.median(vals):10.2f} per unit² (n {len(vals)}), share bending down {np.mean(np.array(vals) < 0):.2f}")


def main(dirs):
    files = sorted(p for d in dirs for p in Path(d).glob("dense-t*.json"))
    rows = rows_of(files)
    if not rows:
        right_vs_wrong(sorted(p for d in dirs for p in Path(d).glob("dense-cur-t*.json")))
        return
    qs = {r["q"] for r in rows}
    print(f"{len(files)} traces over {len(qs)} questions, {len(rows)} sentences; confident (log-odds > 0) at {np.mean([r['conf'] > 0 for r in rows]):.2f} of them; "
          f"robustness capped at the ladder's end for {np.mean([r['capped'] >= 0.5 for r in rows]):.2f}")
    c = [r for r in rows if r["conf"] > 0]
    if c:
        print(f"among confident sentences: Spearman(confidence, robustness) = {spearman([r['conf'] for r in c], [r['rob'] for r in c]):.2f}")
    matched(rows, "written", "whether the answer has been written")
    matched(rows, "kind", "trace kind")
    matched(rows, "reply", "what the model itself replies when stopped")
    g = gap_before_after(rows)
    if g is not None:
        lo, hi = bootstrap(rows, gap_before_after)
        print(f"\nrobustness before minus after the answer is written, at matched confidence: {g:+.3f} (95% over questions: {lo:+.3f} to {hi:+.3f})")

    st = steps(rows)
    exp = expected(rows)
    for s in st:
        s["dres"] = (s["rob"] - exp[band(s["conf"])]) - ((s["rob"] - s["drob"]) - exp[band(s["conf"] - s["dconf"])])
    if st:
        print(f"\n{len(st)} sentence steps; Spearman(Δconfidence, Δrobustness) = {spearman([s['dconf'] for s in st], [s['drob'] for s in st]):.2f}")
    ev = events(st)
    before = [e for e in ev if e["before"]]
    print(f"robustness events (|Δrobustness| ≥ 0.1, |Δconfidence| ≤ 1): {len(ev)}, of which {len(before)} before the answer is written, "
          f"over {len({e['q'] for e in before})} of {len(qs)} questions")
    rate = lambda rs: len([e for e in events(steps(rs)) if e["before"]]) / max(1, len([s for s in steps(rs) if s["before"]]))
    lo, hi = bootstrap(rows, rate)
    print(f"event rate per sentence before the answer: {rate(rows):.3f} (95% over questions: {lo:.3f} to {hi:.3f})")
    print(f"\n{'tag':12s} {'n':>5s} {'Δconfidence':>14s} {'Δrobustness':>14s} {'Δ beyond confidence':>20s} {'events':>7s}")
    for name in [t for t, _ in TAGS] + ["other"]:
        r = [s for s in st if s["tag"] == name]
        if not r:
            continue
        cols = []
        for k in ("dconf", "drob", "dres"):
            v = np.array([x[k] for x in r]); cols.append(f"{v.mean():7.3f} ± {v.std() / math.sqrt(len(v)):5.3f}")
        print(f"{name:12s} {len(r):5d} {cols[0]:>14s} {cols[1]:>14s} {cols[2]:>20s} {sum(abs(x['drob']) >= 0.1 and abs(x['dconf']) <= 1 for x in r):7d}")
    if before:
        print("\nlargest robustness moves before the answer is written, with the sentence read:")
        for e in sorted(before, key=lambda e: -abs(e["drob"]))[:12]:
            print(f"  {e['drob']:+.2f} radius, {e['dconf']:+.1f} nats, at {e['pos']:.2f} of {e['q'][:38]:38s} | {e['text'][:90]}")
    curvature_by_scale(rows)
    right_vs_wrong(sorted(p for d in dirs for p in Path(d).glob("dense-cur-t*.json")))


def right_vs_wrong(files):
    """Whatever number the model gives at each point: is a wrong one more fragile than a right one held as confidently?
    An answer already written in the reasoning is sturdier whether right or wrong, so the comparison is made within
    written and within unwritten answers, then pooled."""
    rows = []
    for p in files:
        d = json.loads(p.read_text())
        for q in d["points"]:
            if q.get("target") is None or q.get("target_log_odds") is None or not q["rays"]:
                continue
            flips = [r["flip"] if r["flip"] is not None else d["radii"][-1] for r in q["rays"]]
            if len(q["target"].lstrip("-")) < 2:
                continue   # a one-digit number turns up everywhere in maths, so whether it is "written" means nothing
            written = written_re(q["target"]).search(d["think"][:q["end"]]) is not None
            rows.append({"q": d["q"], "right": q["target_right"], "conf": q["target_log_odds"], "rob": float(np.median(flips)),
                         "written": written, "pos": q["end"] / max(1, len(d["think"])), "said": q["target"]})
    if not rows:
        return
    wrong = [r for r in rows if not r["right"]]
    print(f"\n== right against wrong: {len(files)} traces, {len(rows)} points where the model gives a number, {len(wrong)} wrong, "
          f"over {len({r['q'] for r in wrong})} questions; wrong and more likely than not: {sum(r['conf'] > 0 for r in wrong)}")
    print(f"{'confidence':12s} {'written, right':>16s} {'wrong':>14s} {'unwritten, right':>18s} {'wrong':>14s}")
    for lo, hi in zip(BINS, BINS[1:]):
        cells = []
        for w in (True, False):
            for ok in (True, False):
                v = [r["rob"] for r in rows if r["written"] == w and r["right"] == ok and lo <= r["conf"] < hi]
                cells.append(f"{np.median(v):.3f} ({len(v):4d})" if v else "—")
        print(f"{band(lo):12s} {cells[0]:>16s} {cells[1]:>14s} {cells[2]:>18s} {cells[3]:>14s}")
    def gap(rs, written=None):
        num = den = 0
        for w in ((True, False) if written is None else (written,)):
            for lo, hi in zip(BINS, BINS[1:]):
                a = [r["rob"] for r in rs if r["written"] == w and r["right"] and lo <= r["conf"] < hi]
                b = [r["rob"] for r in rs if r["written"] == w and not r["right"] and lo <= r["conf"] < hi]
                if len(a) >= 3 and len(b) >= 3:
                    k = min(len(a), len(b)); num += k * (np.median(a) - np.median(b)); den += k
        return num / den if den else None
    for name, w in (("within written and unwritten, pooled", None), ("written only", True), ("unwritten only", False)):
        g = gap(rows, w)
        if g is not None:
            lo, hi = bootstrap(rows, lambda rs: gap(rs, w))
            print(f"right minus wrong robustness at matched confidence, {name}: {g:+.3f} (95% over questions: {lo:+.3f} to {hi:+.3f})")
    for r in sorted(wrong, key=lambda r: -r["conf"])[:6]:
        print(f"  wrong but sure: says {r['said']!r:>11} at log-odds {r['conf']:5.1f}, {'written' if r['written'] else 'unwritten'}, "
              f"robustness {r['rob']:.2f}, at {r['pos']:.2f} of {r['q'][:52]}")


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 2 else math.nan


if __name__ == "__main__":
    main(sys.argv[1:])
