"""How robustly does a reasoning model compute its answer, and how does that change as it reads its own reasoning?

The model (Qwen3-8B) and the question are fixed; the reasoning grows. At a point of the reasoning we stop the model and
ask for the answer, then do the same for copies of it whose weights are pushed along fixed random directions, normalised
row by row to the weights (Li et al. 2018), regenerated from seeds so they never need storing.

    python study.py generate --out OUT --part i --of n       # sample reasoning traces       -> OUT/traces-<i>.json
    python study.py screen --out OUT --part i --of n         # score each at 21 points       -> OUT/screen-<i>.json
    python study.py select --out OUT                         # label and keep them           -> OUT/selection.json
    python study.py surface --out OUT --selection SEL ...    # 51×51 slices for the pictures -> OUT/surface-t*-f*.json
    python study.py dense --out OUT --part i --of n          # per sentence: confidence, and how far along fixed rays in
                                                             # weight space the answer survives -> OUT/dense-t*.json

An answer is scored as its digits followed by any token that starts with "}", with log-probabilities computed exactly
near one (fp32 logits from the output head, float64 log-sums), so log-odds stay accurate when the model is certain.
"""
import argparse
import json
import math
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F

QUESTIONS = [
    {"q": "What is the sum of all positive divisors of 360?", "a": "1170"},
    {"q": "How many positive integers n at most 200 are relatively prime to 200?", "a": "80"},
    {"q": "What is the remainder when 3^47 is divided by 23?", "a": "4"},
    {"q": "How many four-digit positive integers have digits that sum to 9?", "a": "165"},
    {"q": "In how many ways can 8 people sit around a round table if two particular people must not sit next to each other? Rotations count as the same seating.", "a": "3600"},
    {"q": "How many trailing zeros does 125! have?", "a": "31"},
    {"q": "The product of two consecutive positive even integers is 1848. What is their sum?", "a": "86"},
    {"q": "What is the smallest positive integer n such that n! is divisible by 1000?", "a": "15"},
    {"q": "How many integers from 1 to 1000 inclusive are divisible by 3 or 5 but not by 7?", "a": "401"},
    {"q": "What is the sum of the digits of 2^20?", "a": "31"},
    {"q": "How many positive integers less than 500 have exactly three positive divisors?", "a": "8"},
    {"q": "How many subsets of {1, 2, ..., 10} have elements that sum to 10?", "a": "10"},
    {"q": "What is the remainder when 1 + 2 + 3 + ... + 2026 is divided by 9?", "a": "1"},
    {"q": "What is the remainder when 7^100 is divided by 13?", "a": "9"},
    {"q": "What is the units digit of 3^2026?", "a": "9"},
    {"q": "How many primes are there strictly between 80 and 100?", "a": "3"},
    {"q": "What is the remainder when 2^2026 is divided by 7?", "a": "2"},
    {"q": "How many positive integers less than 1000 are divisible by 7 but not by 5?", "a": "114"},
    {"q": "How many ordered pairs of positive integers (x, y) satisfy x*y = 360 and x < y?", "a": "12"},
    {"q": "What is 17 * 23 + 19 * 29?", "a": "942"},
    {"q": "What is the sum of the first 50 odd positive integers?", "a": "2500"},
    {"q": "How many positive divisors does 2^5 * 3^4 * 5 have?", "a": "60"},
    {"q": "What is the remainder when 5^99 is divided by 7?", "a": "6"},
    {"q": "What is the least common multiple of 84 and 90?", "a": "1260"},
    {"q": "In how many distinct ways can the letters of the word BANANA be arranged?", "a": "60"},
    {"q": "What is the sum of the interior angles of a heptagon, in degrees?", "a": "900"},
    {"q": "If 3x + 7 = 2x + 19 and y = 2x - 5, what is y?", "a": "19"},
    {"q": "How many perfect squares are there between 100 and 1000 inclusive?", "a": "22"},
    {"q": "What is the greatest common divisor of 1071 and 462?", "a": "21"},
    {"q": "How many integers from 1 to 100 inclusive are divisible by none of 2, 3 and 5?", "a": "26"},
    {"q": "What is the sum of all integers from 1 to 100 that are divisible by 3 or by 5?", "a": "2418"},
    {"q": "How many solutions in positive integers does a + b + c = 12 have?", "a": "55"},
    {"q": "What is the units digit of 7^2026?", "a": "9"},
    {"q": "A rectangle has perimeter 46 and a diagonal of length 17. What is its area?", "a": "120"},
    {"q": "What is the smallest prime number greater than 200?", "a": "211"},
    {"q": "How many trailing zeros does 1000! have?", "a": "249"},
    {"q": "What is 2^10 + 3^6 + 5^4?", "a": "2378"},
    {"q": "In how many ways can 5 identical balls be placed into 3 distinct boxes?", "a": "21"},
    {"q": "What is the sum of the digits of 3^10?", "a": "27"},
    {"q": "How many three-digit positive integers are divisible by 13?", "a": "69"},
]
# Harder questions, so that some traces end wrong: answers computed in hard.py.
HARD = [
    {"q": 'What is the sum of the digits of 2^100?', "a": "115"},
    {"q": 'What is the sum of the digits of 3^50?', "a": "144"},
    {"q": 'What is the sum of the digits of 7^20?', "a": "76"},
    {"q": 'What is the sum of the digits of 25!?', "a": "72"},
    {"q": 'What is the sum of the digits of 123456789 × 987654321?', "a": "63"},
    {"q": 'What is the sum of the squares of the digits of 2^50?', "a": "514"},
    {"q": 'What is the remainder when 7^2024 is divided by 1000?', "a": "401"},
    {"q": 'What is the remainder when 13^500 is divided by 1000?', "a": "1"},
    {"q": 'What is the remainder when 2^2024 is divided by 1000?', "a": "216"},
    {"q": 'What is the remainder when 2^100 + 3^100 is divided by 37?', "a": "9"},
    {"q": 'What is the remainder when 1^5 + 2^5 + ... + 100^5 is divided by 7?', "a": "5"},
    {"q": 'What is the remainder when 100! is divided by 103?', "a": "51"},
    {"q": 'How many integers from 1 to 10000 are divisible by at least one of 6, 10 and 15?', "a": "2666"},
    {"q": 'How many integers from 1 to 5000 are divisible by 3 or 7 but not by 5?', "a": "1714"},
    {"q": 'How many integers from 1 to 1000 share no common factor greater than 1 with 1001?', "a": "720"},
    {"q": 'How many five-digit positive integers have digits that sum to 20?', "a": "4998"},
    {"q": 'How many positive integers less than 1000 have digits that sum to 15?', "a": "73"},
    {"q": 'In how many ways can 30 be written as an ordered sum of 1s, 2s and 3s?', "a": "53798080"},
    {"q": 'In how many ways can 25 be written as a sum of distinct positive integers, ignoring order?', "a": "142"},
    {"q": 'How many subsets of {1, 2, ..., 12}, including the empty set, have a sum divisible by 4?', "a": "1024"},
    {"q": 'What is the sum of all primes less than 300?', "a": "8275"},
    {"q": 'How many primes are there between 2000 and 2200?', "a": "24"},
    {"q": 'How many positive divisors does 12^10 × 15^5 have?', "a": "2016"},
    {"q": 'What is the sum of all positive divisors of 2024?', "a": "4320"},
    {"q": 'How many zeros does 2024! end with when written in base 12?', "a": "1006"},
    {"q": 'How many points (x, y) with integer coordinates satisfy x^2 + y^2 ≤ 100?', "a": "317"},
    {"q": 'How many solutions in non-negative integers does x + 2y + 3z = 60 have?', "a": "331"},
    {"q": "How many strings of 15 letters, each A or B, contain no three A's in a row?", "a": "10609"},
    {"q": 'What is the floor of 1000 × (√2 + √3)^2?', "a": "9898"},
    {"q": 'How many positive integers n at most 500 make n^2 + n + 1 divisible by 7?', "a": "143"},
    {"q": 'How many binary strings of length 20 have exactly 7 ones, no two of them adjacent?', "a": "3432"},
    {"q": 'What is the coefficient of x^10 in the expansion of (1 + x + x^2)^8?', "a": "784"},
    {"q": 'How many ordered pairs of positive integers (x, y) satisfy 1/x + 1/y = 1/60?', "a": "45"},
    {"q": 'How many triangles with vertices among the vertices of a regular 20-gon are obtuse?', "a": "720"},
    {"q": 'How many integers from 1 to 1000 have exactly five 1s in their binary representation?', "a": "252"},
    {"q": 'How many four-digit positive integers have digits in non-decreasing order that sum to 20?', "a": "33"},
    {"q": 'How many squares of any size, with sides along the grid lines, are there in a 9 by 12 grid of unit squares?', "a": "420"},
    {"q": 'What is 987 × 654 × 321?', "a": "207204858"},
    {"q": 'What is the sum of the digits of 11^12?', "a": "55"},
    {"q": 'How many positive integers less than 2024 are divisible by neither 2, 3 nor 11?', "a": "614"},
]

CUE = "\n</think>\n\nThe final answer is \\boxed{"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load(path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map=DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def prompt_for(tok, q):
    msgs = [{"role": "user", "content": q + "\n\nThink it through, then give the final answer as a single number inside \\boxed{}."}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)


def written_re(x):
    """The number as it might appear in the text, allowing thousands separators and a sentence-ending full stop."""
    d = x.lstrip("-")
    body = "".join(c + ("[, ]?" if i < len(d) - 1 and (len(d) - 1 - i) % 3 == 0 else "") for i, c in enumerate(d))
    return re.compile(rf"(?<![\d.,]){re.escape(x[:len(x) - len(d)])}{body}(?!\d|\.\d|,\d)")


def clean(ans):
    return None if ans is None else ans.replace(" ", "").replace(",", "").replace("$", "").strip()


def write(path, obj):
    tmp = Path(str(path) + ".tmp"); tmp.write_text(json.dumps(obj)); tmp.replace(path)


# ------------------------------------------------------------------------------------------------------------ generate

def generate(args):
    """Sample reasoning traces. By default part i takes the questions i, i+n, ...; with --qids, only those questions, their
    samples spread over the parts and numbered from --sample-offset so they never clash with earlier ones."""
    model, tok = load(args.model)
    tok.padding_side = "left"
    qs = HARD if args.questions == "hard" else QUESTIONS
    if args.qids:
        jobs = [(qi, qs[qi], s) for qi in map(int, args.qids.split(",")) for s in range(args.sample_offset, args.sample_offset + args.samples)]
        jobs = jobs[args.part::args.of]
    else:
        jobs = [(qi, q, s) for qi, q in enumerate(qs) if qi % args.of == args.part for s in range(args.samples)]
    traces = []
    for b in range(0, len(jobs), args.batch):   # one batch at a time, so long traces fit in memory
        batch = jobs[b:b + args.batch]
        enc = tok([prompt_for(tok, q["q"]) for _, q, _ in batch], return_tensors="pt", padding=True).to(DEVICE)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new, do_sample=True, temperature=0.6, top_p=0.95, top_k=20, pad_token_id=tok.pad_token_id)
        log(f"generated {len(batch)} traces in {time.time() - t0:.0f}s")
        for (qi, q, s), ids in zip(batch, out[:, enc.input_ids.shape[1]:]):
            text = tok.decode(ids, skip_special_tokens=True)
            think = text.split("</think>")[0].replace("<think>", "").strip()
            boxed = re.findall(r"\\boxed\{([^{}]*)\}", text.split("</think>")[-1])
            ans = clean(boxed[-1]) if boxed else None
            t = {"qi": qi, "sample": s, "q": q["q"], "a": q["a"], "think": think, "answer": ans, "finished": "</think>" in text,
                 "correct": ans == q["a"], "think_tokens": len(tok(think, add_special_tokens=False).input_ids)}
            traces.append(t)
            log(f"  q{qi}.{s} {'✓' if t['correct'] else '✗'} {'fin' if t['finished'] else 'cut'} {t['think_tokens']:5d} tokens, said {ans!r} (want {q['a']})")
        write(Path(args.out) / f"traces-{args.part}.json", traces)


# ------------------------------------------------------------------------------------------------------------ scoring

def log_mass(z, idx):
    """log of the probability on token(s) idx under float64 logits z, as −softplus(log mass elsewhere − log mass on idx),
    so it stays exact when the probability is within a hair of one."""
    on = torch.zeros_like(z, dtype=torch.bool); on[idx] = True
    return -F.softplus(z[~on].logsumexp(0) - z[on].logsumexp(0)).item()


def log_odds(l):
    """log p − log(1 − p) from an exact log p."""
    return l - math.log(-math.expm1(l)) if l < 0 else math.inf


class Scorer:
    def __init__(self, model, tok):
        self.model, self.tok = model, tok
        self.dev = model.get_input_embeddings().weight.device
        text = [tok.decode([i]) for i in range(len(tok))]
        # the answer may close as "}", "}.", "}\n\n", …: one token each, so the closing brace is scored as any of them
        self.close = torch.tensor([i for i, t in enumerate(text) if t.startswith("}")], device=self.dev)
        # a reply that begins a number: a digit or a sign, perhaps after whitespace in the same token; bare whitespace defers
        self.numeric = torch.tensor([i for i, t in enumerate(text) if re.match(r"\s*([-+]?\d|[-+]$)", t)], device=self.dev)
        self.space = torch.tensor([i for i, t in enumerate(text) if t and not t.strip()], device=self.dev)
        # logits in fp32 from the output head, which is never perturbed: bf16 logits blur the mass left for other answers
        self.head = model.lm_head.weight.float()

    def logits(self, h):
        return F.linear(h.float(), self.head.to(h.device)).double()

    def context(self, q, prefix):
        return self.tok(prompt_for(self.tok, q) + "<think>\n" + prefix + CUE, add_special_tokens=False).input_ids

    def cand_ids(self, answers):
        return [self.tok(a, add_special_tokens=False).input_ids for a in answers]

    @torch.no_grad()
    def logps(self, ctx, cands, starts=False):
        """log p of each candidate answer and then a closing brace after the context: one pass over the context, then each
        answer from its cache. With starts, also log p that the reply begins a number, and that it begins with bare whitespace."""
        out = self.model.model(input_ids=torch.tensor([ctx], device=self.dev), use_cache=True)
        cache, first = out.past_key_values, self.logits(out.last_hidden_state[0, -1])
        res = []
        for c in cands:
            o = self.model.model(input_ids=torch.tensor([c], device=self.dev), past_key_values=cache, use_cache=True)
            z = self.logits(o.last_hidden_state[0])
            res.append(log_mass(first, c[0]) + sum(log_mass(z[i], c[i + 1]) for i in range(len(c) - 1)) + log_mass(z[-1], self.close))
            cache.crop(-len(c))  # back to the end of the context
        return (res, log_mass(first, self.numeric), log_mass(first, self.space)) if starts else res

def margin_of(lps):
    return lps[0] - max(lps[1:]) if len(lps) > 1 else lps[0]



# ------------------------------------------------------------------------------------------------------------ screen

def candidates_for(sc, ts):
    """a*, the other answers the samples gave, the answer given with no reasoning, and a*±1."""
    a = ts[0]["a"]
    ctx = sc.context(ts[0]["q"], "")
    with torch.no_grad():
        g = sc.model.generate(torch.tensor([ctx], device=sc.dev), max_new_tokens=8, do_sample=False, pad_token_id=sc.tok.pad_token_id)
    cold = clean(sc.tok.decode(g[0, len(ctx):], skip_special_tokens=True).split("}")[0])
    cands = [a] + [c for c in dict.fromkeys([t["answer"] for t in ts if t["answer"] and t["answer"] != a] + [cold])
                   if c and c != a and re.fullmatch(r"-?\d+(\.\d+)?", c)]
    if re.fullmatch(r"\d+", a):
        cands += [n for n in (str(int(a) - 1), str(int(a) + 1)) if n not in cands]
    return cands[:8], cold


def screen(args):
    """Score every finished trace at evenly spaced points of its reasoning: cheap, and kept, to search later."""
    out = Path(args.out)
    traces = [t for p in sorted(out.glob("traces-*.json")) for t in json.loads(p.read_text())]
    model, tok = load(args.model)
    sc = Scorer(model, tok)
    by_q = {}
    for t in traces:
        by_q.setdefault(t["qi"], []).append(t)
    res = []
    for qi, ts in sorted(by_q.items()):
        if qi % args.of != args.part:
            continue
        cands, cold = candidates_for(sc, ts)
        cids = sc.cand_ids(cands)
        for t in ts:
            if not t["finished"]:
                continue
            ids = tok(t["think"], add_special_tokens=False).input_ids
            pts = []
            for k in range(args.screen + 1):
                prefix = prefix_at(tok, ids, k / args.screen)
                lps = sc.logps(sc.context(t["q"], prefix), cids)
                pts.append({"k": k, "fraction": k / args.screen, "logps": lps, "margin": margin_of(lps),
                            "answer_written": re.search(rf"(?<![\d.]){re.escape(t['a'])}(?![\d.])", prefix) is not None})
            res.append({**{k: t[k] for k in ("qi", "sample", "q", "a", "answer", "correct", "think_tokens")}, "candidates": cands, "cold": cold, "points": pts})
            m = [p["margin"] for p in pts]
            log(f"q{qi}.{t['sample']} {'✓' if t['correct'] else '✗'} {t['think_tokens']:5d} tok, margin {m[0]:6.2f} → {m[len(m) // 2]:6.2f} → {m[-1]:6.2f}")
        write(out / f"screen-{args.part}.json", res)


# ------------------------------------------------------------------------------------------------------------ select

def select(args):
    """Every finished trace short enough to slice, labelled: resolving (unsure at the start, right at the end), knew (sure
    from the start), wrong (ends on a wrong answer), or other."""
    out = Path(args.out)
    screened = [t for p in sorted(out.glob("screen-*.json")) for t in json.loads(p.read_text())]
    traces = {(t["qi"], t["sample"]): t for p in sorted(out.glob("traces-*.json")) for t in json.loads(p.read_text())}
    fits = [t for t in screened if t["think_tokens"] <= args.max_think]
    start = lambda t: t["points"][0]["margin"]
    end = lambda t: t["points"][-1]["margin"]
    resolving = sorted([t for t in fits if t["correct"] and start(t) < 1.0 and end(t) > 1.0], key=lambda t: -(end(t) - start(t)))
    knew = sorted([t for t in fits if t["correct"] and start(t) > 3.0], key=lambda t: -start(t))
    wrong = [t for t in fits if not t["correct"] and t["answer"]]
    chosen = [(t, "resolving") for t in resolving[:args.resolving]] + [(t, "knew") for t in knew[:args.controls]] + [(t, "wrong") for t in wrong[:args.controls]]
    taken = {(t["qi"], t["sample"]) for t, _ in chosen}
    chosen += [(t, "other") for t in fits if (t["qi"], t["sample"]) not in taken]   # the labels pick examples; trends use every trace
    log(f"{len(screened)} screened: {len(resolving)} resolving, {len(knew)} knew the answer, {len(wrong)} wrong; measuring {len(chosen)}")
    sel = [{**{k: t[k] for k in ("qi", "sample", "q", "a", "answer", "correct", "think_tokens", "candidates", "cold")},
            "kind": kind, "think": traces[(t["qi"], t["sample"])]["think"], "screen": t["points"]} for t, kind in chosen]
    write(out / "selection.json", {"traces": sel})


def say(sc, ctx, n=16):
    """The model's own greedy reply after the context, as text."""
    with torch.no_grad():
        g = sc.model.generate(torch.tensor([ctx], device=sc.dev), max_new_tokens=n, do_sample=False, pad_token_id=sc.tok.pad_token_id)
    return sc.tok.decode(g[0, len(ctx):], skip_special_tokens=True)


def kind_of(reply, a):
    """right: the answer then a closing brace; number: another number; other: anything else, or no closing brace in time."""
    if "}" not in reply:
        return "other"
    ans = clean(reply.split("}")[0])
    return "right" if ans == a else "number" if re.fullmatch(r"[-+]?\d+(\.\d+)?", ans or "") else "other"


def own_answers(sc, tr, n=10):
    """What the model says when stopped at n+1 evenly spaced points of its reasoning and made to answer, greedily."""
    ids, said = sc.tok(tr["think"], add_special_tokens=False).input_ids, []
    for k in range(n + 1):
        said.append(clean(say(sc, sc.context(tr["q"], prefix_at(sc.tok, ids, k / n))).split("}")[0]))
    return said


def full_candidates(sc, tr, cap=12):
    """The right answer, the trace's listed competitors, and every answer the model itself gives when stopped early:
    without the last, most of the probability can sit on answers the margin never sees."""
    said = own_answers(sc, tr)
    extra = [a for a in dict.fromkeys(said) if a and a != tr["a"] and a not in tr["candidates"] and len(a) <= 24]
    return (tr["candidates"] + extra)[:max(cap, len(tr["candidates"]))], said


# ------------------------------------------------------------------------------------------------------------ weights

def prefix_at(tok, ids, frac):
    """The thinking up to a fraction of its tokens, cut back to a sentence end when one is close."""
    if frac <= 0:
        return ""
    text = tok.decode(ids[: round(len(ids) * frac)])
    if frac < 1:
        m = max(text.rfind(".\n"), text.rfind(". "), text.rfind("\n\n"))
        if m > len(text) * 0.8:
            text = text[: m + 1]
    return text.strip()


class Weights:
    """Every weight matrix of the transformer layers, seeded row-normalised directions, and perturbation along them."""

    def __init__(self, model):
        self.all = [p for layer in model.model.layers for p in layer.parameters() if p.ndim == 2]
        self.orig, self.planes = None, {}

    @staticmethod
    def direction(p, seed, j):
        g = torch.Generator(device=p.device).manual_seed(seed * 100003 + j)
        r = torch.randn(p.shape, generator=g, device=p.device, dtype=torch.float32)
        return r * (p.float().norm(dim=1, keepdim=True) / r.norm(dim=1, keepdim=True).clamp_min(1e-12))

    def hold(self, seeds=()):
        with torch.no_grad():
            self.orig = [p.detach().clone() for p in self.all]
            self.planes = {s: [self.direction(p, s, j).to(torch.bfloat16) for j, p in enumerate(self.all)] for s in seeds}

    def set(self, coeffs):
        with torch.no_grad():
            for j, (p, o) in enumerate(zip(self.all, self.orig)):
                w = o.float()
                for seed, c in coeffs.items():
                    w += c * (self.planes[seed][j].float() if seed in self.planes else self.direction(p, seed, j))
                p.copy_(w.to(p.dtype))

    def release(self):
        self.set({})
        self.orig, self.planes = None, {}
        if DEVICE == "cuda": torch.cuda.empty_cache()


def surface(args):
    """Fine slices for the pictures over two fixed directions (--seeds), at chosen points of chosen traces: −log p(a*), the
    margin, the log-odds, every candidate's log p, log p of beginning a number; and, on an 11×11 subgrid, what each copy
    actually writes. One (trace, point) per job slot."""
    out = Path(args.out)
    sel = json.loads(Path(args.selection).read_text())
    traces = [int(t) for t in args.trace_ids.split(",")] if args.trace_ids else list(range(len(sel["traces"])))
    fracs = [float(f) for f in args.fractions.split(",")]
    jobs = [(ti, f) for ti in traces for f in fracs][args.part::args.of]
    if not jobs:
        log("nothing to do"); return
    model, tok = load(args.model)
    sc, wts = Scorer(model, tok), Weights(model)
    grid = torch.linspace(-args.span, args.span, args.grid).tolist()
    n, seeds = len(grid), tuple(int(x) for x in args.seeds.split(","))
    sub = list(range(n)) if args.replies == "all" else list(range(0, n, max(1, (n - 1) // 10)))   # whose reply to record
    keys = ("loss", "margin", "log_odds", "logps", "numeric", "space")
    for ti, f in jobs:
        t0 = time.time()
        tr = sel["traces"][ti]
        prefix = prefix_at(tok, tok(tr["think"], add_special_tokens=False).input_ids, f)
        cand_list, said = full_candidates(sc, tr)
        ctx, cands = sc.context(tr["q"], prefix), sc.cand_ids(cand_list)
        wts.hold(seeds=seeds)
        G = {k: [[None] * n for _ in grid] for k in keys}
        for i, a in enumerate(grid):
            for j, b in enumerate(grid):
                wts.set({seeds[0]: a, seeds[1]: b}); lps, num, sp = sc.logps(ctx, cands, starts=True)
                for k, v in zip(keys, (-lps[0], margin_of(lps), log_odds(lps[0]), [float(f"{l:.6g}") for l in lps], num, sp)):
                    G[k][i][j] = v
        replies = [[None] * len(sub) for _ in sub]
        for ii, i in enumerate(sub):
            for jj, j in enumerate(sub):
                wts.set({seeds[0]: grid[i], seeds[1]: grid[j]}); replies[ii][jj] = say(sc, ctx)
        wts.release()
        plane = "" if seeds == (300, 301) else f"-p{seeds[0]}"
        write(out / f"surface-t{ti}-f{f:.3f}-g{args.grid}-s{args.span}{plane}.json",
              {"trace": ti, "fraction": f, "q": tr["q"], "a": tr["a"], "seeds": seeds, "candidates": cand_list, "said_when_stopped": said,
               "context_tokens": len(ctx), "prefix_tail": prefix[-500:], "grid": grid, **G,
               "replies": {"index": sub, "text": replies, "kind": [[kind_of(r, tr["a"]) for r in row] for row in replies]}})
        L, M, c = G["loss"], G["margin"], n // 2
        log(f"trace {ti} at {f:.3f}: loss at centre {L[c][c]:.2f}, range {min(map(min, L)):.2f}–{max(map(max, L)):.2f}; "
            f"margin {M[c][c]:.2f}, range {min(map(min, M)):.2f}–{max(map(max, M)):.2f} ({time.time() - t0:.0f}s)")


# ------------------------------------------------------------------------------------------------------------ dense

def sentence_ends(think, min_chars=40, cap=160):
    """Character positions where a sentence or line ends, merging short ones, at most `cap` of them."""
    ends, last = [0], 0
    for m in re.finditer(r"(?<=[.?!:])\s+|\n+", think):
        if m.start() - last >= min_chars:
            ends.append(m.start()); last = m.start()
    if len(think) - last > 0:
        ends.append(len(think))
    if len(ends) > cap:
        idx = sorted(set(round(i * (len(ends) - 1) / (cap - 1)) for i in range(cap)))
        ends = [ends[i] for i in idx]
    return ends


def flip_radius(o0, radii, lo):
    """Where the right answer stops being more likely than not along a ray: 0 if it already is not at the model, the
    interpolated crossing of log-odds zero, or None if it survives the whole ladder."""
    if o0 <= 0:
        return 0.0
    pr, po = 0.0, o0
    for r, o in zip(radii, lo):
        if o <= 0:
            return pr + (r - pr) * po / (po - o)
        pr, po = r, o
    return None


def dense(args):
    """At every sentence boundary of each trace: the answer's scores at θ, then along fixed weight directions, both ways, at
    a ladder of radii, stopping once the right answer is gone. From these: confidence (log-odds at θ), the radius at
    which the answer flips along each ray, and the bending at every scale of the ladder."""
    out = Path(args.out)
    sel = json.loads(Path(args.selection or out / "selection.json").read_text())
    pool = [ti for ti in range(min(len(sel["traces"]), args.trace_limit)) if ti % args.stride == 0]
    mine = pool[args.part::args.of][::-1 if args.reverse else 1]   # --reverse: a second worker on the same share, from the other end
    if not mine:
        log("nothing to do"); return
    model, tok = load(args.model)
    sc, wts = Scorer(model, tok), Weights(model)
    seeds = [int(x) for x in args.directions.split(",")]
    radii = [float(x) for x in args.radii.split(",")]
    for ti in mine:
        path = out / {"right": f"dense-t{ti}.json", "own": f"dense-own-t{ti}.json", "current": f"dense-cur-t{ti}.json"}[args.target]
        if path.exists():
            continue
        t0 = time.time()
        tr = {**sel["traces"][ti], "right": sel["traces"][ti]["a"]}
        if args.target == "own":
            if not tr.get("answer"):
                continue
            # the answer the model ends up giving, right or wrong, measured exactly as the right one would be
            tr = {**tr, "a": tr["answer"], "candidates": [tr["answer"]] + [c for c in tr["candidates"] if c != tr["answer"]]}
        ends = sentence_ends(tr["think"], cap=args.max_sentences)
        ctxs = [sc.context(tr["q"], tr["think"][:e]) for e in ends]
        cand_list, said = full_candidates(sc, tr)
        cands = sc.cand_ids(cand_list)
        at = [sc.logps(c, cands, starts=True) for c in ctxs]
        replies = [say(sc, c) for c in ctxs]
        if args.target == "current":
            # at each boundary, whatever number the model answers right then, followed along the rays as the right one would be
            now = [clean(r.split("}")[0]) if "}" in r else None for r in replies]
            now = [a if a and re.fullmatch(r"[-+]?\d+", a) else None for a in now]
            tgt = [sc.cand_ids([a])[0] if a else None for a in now]
            o_at = [log_odds(sc.logps(c, [t])[0]) if t else None for c, t in zip(ctxs, tgt)]
        else:
            now, tgt, o_at = [tr["a"]] * len(ctxs), [cands[0]] * len(ctxs), [log_odds(a[0][0]) for a in at]
        rays = [[{} for _ in seeds] for _ in ctxs]
        for k0 in range(0, len(seeds), 2):  # two directions held at a time
            pair = seeds[k0:k0 + 2]
            wts.hold(seeds=tuple(pair))
            for j, sd in enumerate(pair):
                for i, c in enumerate(ctxs):
                    if tgt[i] is None:
                        continue
                    for sign in (1, -1):
                        row = []
                        for r in radii:   # only the target answer along the rays: the margin is a quantity of θ itself
                            wts.set({sd: sign * r}); lps, num, sp = sc.logps(c, [tgt[i]], starts=True)
                            row.append({"r": r, "logp": lps[0], "log_odds": log_odds(lps[0]), "numeric": num, "space": sp})
                            if row[-1]["log_odds"] < args.gone:
                                break
                        rays[i][k0 + j]["+" if sign > 0 else "-"] = row
            wts.release()
        pts = []
        for i, e in enumerate(ends):
            lps, num, sp = at[i]
            o0 = o_at[i]
            rr = []
            for k, sd in enumerate(seeds):
                for sign in ("+", "-"):
                    if sign in rays[i][k]:
                        row = rays[i][k][sign]
                        rr.append({"seed": sd, "sign": sign, "flip": flip_radius(o0, radii, [x["log_odds"] for x in row]), "ladder": row})
            pts.append({"end": e, "tokens": len(ctxs[i]), "logps": lps, "p": math.exp(lps[0]), "margin": margin_of(lps), "log_odds": log_odds(lps[0]),
                        "target": now[i], "target_right": now[i] == tr["right"], "target_log_odds": o0,
                        "numeric": num, "space": sp, "reply": replies[i], "reply_kind": kind_of(replies[i], tr["a"]),
                        "answer_written": written_re(tr["a"]).search(tr["think"][:e]) is not None,
                        "rays": rr})
        write(path, {"trace": ti, "q": tr["q"], "a": tr["a"], "right": tr["right"], "correct": tr["answer"] == tr["right"], "target": args.target,
                     "kind": tr["kind"], "answer": tr["answer"], "candidates": cand_list, "said_when_stopped": said,
                     "think": tr["think"], "directions": seeds, "radii": radii, "points": pts})
        fl = lambda p: sorted((x["flip"] if x["flip"] is not None else radii[-1]) for x in p["rays"])[len(p["rays"]) // 2] if p["rays"] else math.nan
        log(f"trace {ti} ({tr['kind']}): {len(pts)} sentences, log-odds {pts[0]['log_odds']:.1f} → {pts[-1]['log_odds']:.1f}, "
            f"median flip radius {fl(pts[0]):.2f} → {fl(pts[-1]):.2f} ({time.time() - t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["generate", "screen", "select", "surface", "dense"])
    ap.add_argument("--model", default="/root/qwen3-8b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--questions", default="base", choices=["base", "hard"], help="generate: which question set")
    ap.add_argument("--qids", default=None, help="generate: only these questions (comma-separated indices)")
    ap.add_argument("--sample-offset", type=int, default=0, help="generate: number the samples from here")
    ap.add_argument("--batch", type=int, default=32, help="generate: traces per batch")
    ap.add_argument("--max-new", type=int, default=4500)
    ap.add_argument("--max-think", type=int, default=3600)
    ap.add_argument("--resolving", type=int, default=80, help="select: how many resolving traces to label as such")
    ap.add_argument("--controls", type=int, default=15, help="select: how many of each control kind")
    ap.add_argument("--selection", default=None, help="surface: the selection.json whose traces to slice")
    ap.add_argument("--trace-ids", default=None, help="surface: comma-separated trace indices (default all)")
    ap.add_argument("--fractions", default="0,0.5,1", help="surface: comma-separated points of the reasoning")
    ap.add_argument("--directions", default="300,301,500,501,502,503", help="dense: seeds of the rays; the first two are the plane of the pictures")
    ap.add_argument("--radii", default="0.03,0.06,0.12,0.2,0.3,0.45", help="dense: the ladder of radii along each ray")
    ap.add_argument("--gone", type=float, default=-8.0, help="dense: stop a ray once the log-odds fall below this")
    ap.add_argument("--stride", type=int, default=1, help="dense: only every stride-th trace")
    ap.add_argument("--reverse", action="store_true", help="dense: take this part's traces last to first, skipping ones already done")
    ap.add_argument("--max-sentences", type=int, default=160, help="dense: at most this many boundaries per trace")
    ap.add_argument("--target", default="right", choices=["right", "own", "current"],
                    help="dense: whose robustness: the right answer's, the model's final answer's, or whatever it answers at each point")
    ap.add_argument("--seeds", default="300,301", help="surface: the two directions of the plane")
    ap.add_argument("--replies", default="subgrid", choices=["subgrid", "all"], help="surface: record every copy's reply, or an 11×11 subgrid's")
    ap.add_argument("--trace-limit", type=int, default=10**9, help="dense: only traces with a lower index")
    ap.add_argument("--screen", type=int, default=20, help="screen: points per trace, after the start")
    ap.add_argument("--grid", type=int, default=51, help="surface: points along each direction")
    ap.add_argument("--span", type=float, default=0.45, help="surface: how far along each direction, relative to the weights")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    {"generate": generate, "screen": screen, "select": select, "surface": surface, "dense": dense}[args.cmd](args)


if __name__ == "__main__":
    main()
