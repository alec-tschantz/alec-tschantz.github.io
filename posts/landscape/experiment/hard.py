"""Harder questions for the study, each answer computed here rather than written by hand.

    python3 hard.py          # prints the list as Python, to paste into study.py as HARD

Chosen for what a small model tends to get wrong: long exact multiplication, inclusion-exclusion, counting under
constraints, large modular powers. Every answer is a single integer.
"""
import itertools
import math
from functools import lru_cache


def digits(n):
    return sum(int(c) for c in str(n))


def primes(lo, hi):
    return [p for p in range(max(2, lo), hi) if all(p % d for d in range(2, math.isqrt(p) + 1))]


def ndiv(n):
    return sum(1 for d in range(1, math.isqrt(n) + 1) if n % d == 0 for _ in ((d,) if d * d == n else (d, n // d)))


def sigma(n):
    return sum(d + (n // d if d * d != n else 0) for d in range(1, math.isqrt(n) + 1) if n % d == 0)


def partitions(n, distinct=False):
    @lru_cache(None)
    def p(n, k):  # partitions of n into parts ≤ k (distinct: parts < k+1 each used at most once)
        if n == 0:
            return 1
        if k == 0:
            return 0
        return p(n, k - 1) + (p(n - k, k - 1 if distinct else k) if n >= k else 0)
    return p(n, n)


def compositions_123(n):
    a = [1, 1, 2]
    for _ in range(3, n + 1):
        a.append(a[-1] + a[-2] + a[-3])
    return a[n]


def no_three_a(n):
    return sum(1 for s in itertools.product("AB", repeat=n) if "AAA" not in "".join(s))


def v(n, p):
    k = 0
    while n % p == 0:
        n //= p; k += 1
    return k


def obtuse_20gon():
    pts = [(math.cos(2 * math.pi * k / 20), math.sin(2 * math.pi * k / 20)) for k in range(20)]
    count = 0
    for a, b, c in itertools.combinations(range(20), 3):
        P = [pts[a], pts[b], pts[c]]
        for i in range(3):
            u, w = P[(i + 1) % 3], P[(i + 2) % 3]
            dot = (u[0] - P[i][0]) * (w[0] - P[i][0]) + (u[1] - P[i][1]) * (w[1] - P[i][1])
            if dot < -1e-9:
                count += 1; break
    return count


def poly_coeff(base, power, k):
    c = [1]
    for _ in range(power):
        out = [0] * (len(c) + len(base) - 1)
        for i, x in enumerate(c):
            for j, y in enumerate(base):
                out[i + j] += x * y
        c = out
    return c[k]


Q = [
    ("What is the sum of the digits of 2^100?", digits(2 ** 100)),
    ("What is the sum of the digits of 3^50?", digits(3 ** 50)),
    ("What is the sum of the digits of 7^20?", digits(7 ** 20)),
    ("What is the sum of the digits of 25!?", digits(math.factorial(25))),
    ("What is the sum of the digits of 123456789 × 987654321?", digits(123456789 * 987654321)),
    ("What is the sum of the squares of the digits of 2^50?", sum(int(c) ** 2 for c in str(2 ** 50))),
    ("What is the remainder when 7^2024 is divided by 1000?", pow(7, 2024, 1000)),
    ("What is the remainder when 13^500 is divided by 1000?", pow(13, 500, 1000)),
    ("What is the remainder when 2^2024 is divided by 1000?", pow(2, 2024, 1000)),
    ("What is the remainder when 2^100 + 3^100 is divided by 37?", (pow(2, 100, 37) + pow(3, 100, 37)) % 37),
    ("What is the remainder when 1^5 + 2^5 + ... + 100^5 is divided by 7?", sum(k ** 5 for k in range(1, 101)) % 7),
    ("What is the remainder when 100! is divided by 103?", math.factorial(100) % 103),
    ("How many integers from 1 to 10000 are divisible by at least one of 6, 10 and 15?", sum(1 for n in range(1, 10001) if n % 6 == 0 or n % 10 == 0 or n % 15 == 0)),
    ("How many integers from 1 to 5000 are divisible by 3 or 7 but not by 5?", sum(1 for n in range(1, 5001) if (n % 3 == 0 or n % 7 == 0) and n % 5)),
    ("How many integers from 1 to 1000 share no common factor greater than 1 with 1001?", sum(1 for n in range(1, 1001) if math.gcd(n, 1001) == 1)),
    ("How many five-digit positive integers have digits that sum to 20?", sum(1 for n in range(10000, 100000) if digits(n) == 20)),
    ("How many positive integers less than 1000 have digits that sum to 15?", sum(1 for n in range(1, 1000) if digits(n) == 15)),
    ("In how many ways can 30 be written as an ordered sum of 1s, 2s and 3s?", compositions_123(30)),
    ("In how many ways can 25 be written as a sum of distinct positive integers, ignoring order?", partitions(25, distinct=True)),
    ("How many subsets of {1, 2, ..., 12}, including the empty set, have a sum divisible by 4?", sum(1 for k in range(13) for s in itertools.combinations(range(1, 13), k) if sum(s) % 4 == 0)),
    ("What is the sum of all primes less than 300?", sum(primes(2, 300))),
    ("How many primes are there between 2000 and 2200?", len(primes(2000, 2200))),
    ("How many positive divisors does 12^10 × 15^5 have?", ndiv(12 ** 10 * 15 ** 5)),
    ("What is the sum of all positive divisors of 2024?", sigma(2024)),
    ("How many zeros does 2024! end with when written in base 12?", min(sum(v(k, 2) for k in range(1, 2025)) // 2, sum(v(k, 3) for k in range(1, 2025)))),
    ("How many points (x, y) with integer coordinates satisfy x^2 + y^2 ≤ 100?", sum(1 for x in range(-10, 11) for y in range(-10, 11) if x * x + y * y <= 100)),
    ("How many solutions in non-negative integers does x + 2y + 3z = 60 have?", sum(1 for y in range(31) for z in range(21) if 2 * y + 3 * z <= 60)),
    ("How many strings of 15 letters, each A or B, contain no three A's in a row?", no_three_a(15)),
    ("What is the floor of 1000 × (√2 + √3)^2?", math.floor(1000 * (5 + 2 * math.sqrt(6)))),
    ("How many positive integers n at most 500 make n^2 + n + 1 divisible by 7?", sum(1 for n in range(1, 501) if (n * n + n + 1) % 7 == 0)),
    ("How many binary strings of length 20 have exactly 7 ones, no two of them adjacent?", sum(1 for s in itertools.combinations(range(20), 7) if all(b - a > 1 for a, b in zip(s, s[1:])))),
    ("What is the coefficient of x^10 in the expansion of (1 + x + x^2)^8?", poly_coeff([1, 1, 1], 8, 10)),
    ("How many ordered pairs of positive integers (x, y) satisfy 1/x + 1/y = 1/60?", sum(1 for x in range(61, 3661) if (60 * x) % (x - 60) == 0)),
    ("How many triangles with vertices among the vertices of a regular 20-gon are obtuse?", obtuse_20gon()),
    ("How many integers from 1 to 1000 have exactly five 1s in their binary representation?", sum(1 for n in range(1, 1001) if bin(n).count("1") == 5)),
    ("How many four-digit positive integers have digits in non-decreasing order that sum to 20?", sum(1 for n in range(1000, 10000) if list(str(n)) == sorted(str(n)) and digits(n) == 20)),
    ("How many squares of any size, with sides along the grid lines, are there in a 9 by 12 grid of unit squares?", sum((10 - k) * (13 - k) for k in range(1, 10))),
    ("What is 987 × 654 × 321?", 987 * 654 * 321),
    ("What is the sum of the digits of 11^12?", digits(11 ** 12)),
    ("How many positive integers less than 2024 are divisible by neither 2, 3 nor 11?", sum(1 for n in range(1, 2024) if n % 2 and n % 3 and n % 11)),
]

if __name__ == "__main__":
    print("HARD = [")
    for q, a in Q:
        print(f'    {{"q": {q!r}, "a": "{a}"}},')
    print("]")
