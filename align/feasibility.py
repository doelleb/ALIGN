"""Model-free agreement bounds (Proposition 1) and the achievable-alignment region (Appendix E).

Items are grouped into pattern classes p in {0,1}^K of group-majority labels. A verdict vector is
summarised by t_p in [0, n_p], the number of items of pattern p flagged unsafe, so group d's
agreement A_d = (1/N) sum_p [t_p p_d + (n_p - t_p)(1 - p_d)] is linear in t.
"""
import itertools
from collections import Counter

import numpy as np
from scipy.optimize import linprog


def pairwise_disagreement(maj):
    """Fraction of items on which each pair of group majorities differ (delta in Prop. 1)."""
    maj = maj.dropna()
    return {(a, b): float((maj[a] != maj[b]).mean())
            for a, b in itertools.combinations(maj.columns, 2)}


class PatternLP:
    def __init__(self, maj):
        maj = maj.dropna()
        counts = Counter(tuple(int(x) for x in row) for row in maj.to_numpy())
        self.patterns = list(counts)
        self.P = np.array(self.patterns, float)          # m x K
        self.n = np.array([counts[p] for p in self.patterns], float)
        self.N = self.n.sum()
        self.m, self.K = self.P.shape
        # A_d(t) = slope[:, d] @ t + offset[d]
        self.slope = (2 * self.P - 1) / self.N
        self.offset = (self.n[:, None] * (1 - self.P)).sum(0) / self.N

    def agreement(self, t):
        return self.slope.T @ t + self.offset

    def egalitarian(self):
        """max_t min_d A_d."""
        c = np.zeros(self.m + 1); c[-1] = -1
        A_ub = np.hstack([-self.slope.T, np.ones((self.K, 1))])
        r = linprog(c, A_ub=A_ub, b_ub=self.offset,
                    bounds=[(0, x) for x in self.n] + [(None, None)], method="highs")
        return self.agreement(r.x[:self.m])

    def utilitarian(self):
        """max_t mean_d A_d: the majority-of-groups verdict within each pattern class."""
        share = self.P.mean(1)
        t = np.where(share > .5, self.n, np.where(share == .5, self.n / 2, 0))
        return self.agreement(t)

    def utilitarian_equal(self):
        """max_t mean_d A_d subject to A_1 = ... = A_K."""
        c = -self.slope.mean(1)
        A_eq = (self.slope[:, 1:] - self.slope[:, [0]]).T
        b_eq = self.offset[0] - self.offset[1:]
        r = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=[(0, x) for x in self.n], method="highs")
        return self.agreement(r.x)

    def coverage_frontier(self, coverage):
        """Best min-group agreement on covered items when abstaining on a (1 - coverage) share.

        Variables: t_p (flagged), a_p (abstained), z. Abstained items are taken from the
        unflagged ones without loss of generality.
        """
        m, N = self.m, self.N
        c = np.zeros(2 * m + 1); c[-1] = -1
        A_ub, b_ub = [], []
        for d in range(self.K):
            row = np.zeros(2 * m + 1)
            row[:m] = -(2 * self.P[:, d] - 1)
            row[m:2 * m] = 1 - self.P[:, d]
            row[-1] = N * coverage
            A_ub.append(row); b_ub.append((self.n * (1 - self.P[:, d])).sum())
        for i in range(m):
            row = np.zeros(2 * m + 1); row[i] = row[m + i] = 1
            A_ub.append(row); b_ub.append(self.n[i])
        A_eq = [np.concatenate([np.zeros(m), np.ones(m), [0]])]
        b_eq = [N * (1 - coverage)]
        bounds = [(0, x) for x in self.n] * 2 + [(None, None)]
        r = linprog(c, A_ub=np.array(A_ub), b_ub=b_ub, A_eq=np.array(A_eq), b_eq=b_eq,
                    bounds=bounds, method="highs")
        return float(-r.fun) if r.success else np.nan
