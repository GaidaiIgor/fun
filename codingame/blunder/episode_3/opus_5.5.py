"""Guesses the computational complexity of a program from its measured execution times."""
from math import log
from statistics import pvariance

samples = [tuple(map(int, input().split())) for _ in range(int(input()))]
# Logarithms of the candidate complexity functions, kept in log space so that 2^n does not overflow
CANDIDATES = {"O(1)": lambda n: 0, "O(log n)": lambda n: log(log(n)), "O(n)": log, "O(n log n)": lambda n: log(n) + log(log(n)),
              "O(n^2)": lambda n: 2 * log(n), "O(n^2 log n)": lambda n: 2 * log(n) + log(log(n)), "O(n^3)": lambda n: 3 * log(n),
              "O(2^n)": lambda n: n * log(2)}
# The right complexity makes t / f(n) nearly constant, i.e. log t - log f(n) has the smallest spread
print(min(CANDIDATES, key=lambda name: pvariance([log(t) - CANDIDATES[name](n) for n, t in samples])))
