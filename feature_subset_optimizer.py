#!/usr/bin/env python3
"""
Feature Subset Optimizer for segmented metrics (FTDs & ILS_L2FTD)
================================================================

Goal
----
Find a (near-)minimal subset of categorical dimensions that yields the
best-performing segment(s) with respect to two metrics:
  - FTDs (count)
  - ILS_L2FTD := FTDs / ils_count (conversion ratio)

Approach
--------
1) Greedy forward selection baseline (fast, local optimum).
2) Genetic Algorithm (GA) search over feature subsets (global, robust).

For any candidate subset of dimensions S, we:
  - groupby S and aggregate: ftds_sum, ils_count_sum, accounts_sum
  - compute ils_l2ftd_agg = ftds_sum / ils_count_sum
  - compute a scalar score to balance FTDs and LS ratio
  - take the single best group (segment) under S as the subset's fitness

The GA evolves bitstrings (length = number of usable dimensions) where
1 means the dimension is included in S and 0 means excluded.

Notes
-----
- You can optionally enforce minimum support thresholds to avoid tiny
  segments with high ratios due to small denominators.
- You can optionally add a penalty per feature to encourage smaller subsets.

Usage
-----
python feature_subset_optimizer.py \
  --data datasources/group.parquet \
  --generations 120 --population 80 \
  --w1 1.0 --w2 1.0 \
  --min-ls 1 --min-ftds 1 --min-accounts 1 \
  --penalty-per-feature 0.0 \
  --random-seed 42

Outputs
-------
- Prints summary of dimensions considered
- Greedy best subset + segment and metrics
- GA best subset + segment and metrics
- Top-N GA solutions with their best segments

Requires
--------
Python 3.9+
Packages: pandas, numpy, pyarrow (to read parquet)

"""
from __future__ import annotations
import argparse
import itertools
import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ----------------------------- Config / Constants ---------------------------- #

# Full list as provided in the brief
ALL_DIMENSIONS: List[str] = [
    "cl_country_name",
    "initial_platform",
    "account_license",
    # "account_status",
    # "initial_lead_status",
    # "lead_status",
    "age_group",
    "annual_income",
    "savings",
    "knowledge_of_trading",
    "os",
    "sms_verification",
    # "demo_trade_flag",
    "self_reg_real",
    # "dummy",
]

METRIC_COLS = {
    "ftds": "ftds",
    "ils_count": "ils_count",
    "accounts": "accounts",
}

# ----------------------------- Data Structures ------------------------------ #

@dataclass
class SegmentMetrics:
    key: Tuple  # values for the group key (ordered like dims)
    dims: Tuple[str, ...]  # dims used to group
    ftds: int
    ils_count: int
    accounts: int
    ils_l2ftd: float
    score: float

@dataclass
class SubsetResult:
    dims: Tuple[str, ...]
    segment: SegmentMetrics
    feature_count: int

# ----------------------------- Helper Functions ----------------------------- #

def safe_ratio(num: float, den: float, eps: float = 1e-9) -> float:
    if den <= 0:
        return 0.0
    return float(num) / float(den)


def to_tuple(x: Iterable) -> Tuple:
    return tuple(x)


# ----------------------------- Evaluator with Cache ------------------------- #

class SubsetEvaluator:
    """Evaluates feature subsets against the dataset with caching of groupbys."""

    def __init__(
        self,
        df: pd.DataFrame,
        usable_dims: Sequence[str],
        w1: float = 1.0,
        w2: float = 1.0,
        min_ftds: int = 1,
        min_ils_count: int = 1,
        min_accounts: int = 1,
        penalty_per_feature: float = 0.0,
        eps: float = 1e-9,
    ) -> None:
        self.df = df
        self.usable_dims = list(usable_dims)
        self.w1 = float(w1)
        self.w2 = float(w2)
        self.min_ftds = int(min_ftds)
        self.min_lsc = int(min_ils_count)
        self.min_accounts = int(min_accounts)
        self.penalty_per_feature = float(penalty_per_feature)
        self.eps = float(eps)

        # Cache: map dims tuple -> aggregated dataframe
        self._agg_cache: Dict[Tuple[str, ...], pd.DataFrame] = {}

        # Precompute global sums (for empty subset)
        self._global_agg = (
            self.df[[METRIC_COLS["ftds"], METRIC_COLS["ils_count"], METRIC_COLS["accounts"]]]
            .sum(numeric_only=True)
            .to_dict()
        )

    def _aggregate_for_dims(self, dims: Tuple[str, ...]) -> pd.DataFrame:
        """Group df by dims and aggregate metrics; cached for reuse.
        Returns a dataframe with columns: dims..., ftds, ils_count, accounts, ils_l2ftd
        """
        dims = tuple(dims)
        if dims in self._agg_cache:
            return self._agg_cache[dims]

        if len(dims) == 0:
            # Create a single-row aggregate
            data = {
                "ftds": int(self._global_agg[METRIC_COLS["ftds"]]),
                "ils_count": int(self._global_agg[METRIC_COLS["ils_count"]]),
                "accounts": int(self._global_agg[METRIC_COLS["accounts"]]),
            }
            agg = pd.DataFrame([data])
            agg["ils_l2ftd"] = agg["ftds"].astype(float) / np.where(
                agg["ils_count"] > 0, agg["ils_count"], np.inf
            )
        else:
            agg = (
                self.df.groupby(list(dims), dropna=False)[
                    [METRIC_COLS["ftds"], METRIC_COLS["ils_count"], METRIC_COLS["accounts"]]
                ]
                .sum(numeric_only=True)
                .rename(columns=METRIC_COLS)
                .reset_index()
            )
            agg["ils_l2ftd"] = agg["ftds"].astype(float) / np.where(
                agg["ils_count"] > 0, agg["ils_count"], np.inf
            )

        self._agg_cache[dims] = agg
        return agg

    def _segment_score(self, ftds: float, ils_l2ftd: float, k_features: int) -> float:
        # Weighted log-product style utility. Higher is better.
        # Avoid log(0) via eps.
        val = self.w1 * math.log1p(max(ftds, 0.0)) + self.w2 * math.log(max(ils_l2ftd, self.eps))
        # Optional size penalty
        val -= self.penalty_per_feature * float(k_features)
        return val

    def best_segment_for_dims(self, dims: Sequence[str]) -> Optional[SegmentMetrics]:
        dims_t = tuple(dims)
        agg = self._aggregate_for_dims(dims_t)

        # Apply minimum support filters
        mask = (
            (agg["ftds"] >= self.min_ftds)
            & (agg["ils_count"] >= self.min_lsc)
            & (agg["accounts"] >= self.min_accounts)
        )
        cand = agg.loc[mask].copy()
        if cand.empty:
            return None

        # >>> NEW: exclude groups where any grouping dim is empty/sentinel
        if len(dims_t) > 0 and not cand.empty:
            valid_mask = np.ones(len(cand), dtype=bool)
            for d in dims_t:
                col = cand[d]
                # string-like -> reject empty ""
                if pd.api.types.is_string_dtype(col.dtype) or isinstance(col.dtype, pd.StringDtype):
                    valid_mask &= (col != "")
                # numeric-like -> reject sentinel -9999
                elif pd.api.types.is_numeric_dtype(col.dtype):
                    valid_mask &= (col != -9999)
            cand = cand.loc[valid_mask]
        # <<< END NEW

        if cand.empty:
            return None

        # Compute scores and pick the best row
        k = len(dims_t)
        scores = [self._segment_score(r.ftds, r.ils_l2ftd, k) for r in cand.itertuples(index=False)]
        idx = int(np.argmax(scores))
        best = cand.iloc[idx]

        # Extract key (group values) in the order of dims
        if k == 0:
            key_vals: Tuple = tuple()
        else:
            key_vals = tuple(best[d] for d in dims_t)

        return SegmentMetrics(
            key=key_vals,
            dims=dims_t,
            ftds=int(best["ftds"]),
            ils_count=int(best["ils_count"]),
            accounts=int(best["accounts"]),
            ils_l2ftd=float(best["ils_l2ftd"]),
            score=float(scores[idx]),
        )


# ----------------------------- Greedy Forward Search ------------------------ #

def greedy_forward(
    evaluator: SubsetEvaluator,
    dims_pool: Sequence[str],
) -> Optional[SubsetResult]:
    selected: List[str] = []
    best_seg: Optional[SegmentMetrics] = evaluator.best_segment_for_dims(selected)
    improved = True

    while improved:
        improved = False
        best_add_dim = None
        best_add_seg: Optional[SegmentMetrics] = None

        remaining = [d for d in dims_pool if d not in selected]
        for d in remaining:
            seg = evaluator.best_segment_for_dims(selected + [d])
            if seg is None:
                continue
            if (best_add_seg is None) or (seg.score > best_add_seg.score):
                best_add_seg = seg
                best_add_dim = d

        # Compare against current best
        if best_add_seg is not None and (best_seg is None or best_add_seg.score > best_seg.score):
            selected.append(best_add_dim)  # type: ignore[arg-type]
            best_seg = best_add_seg
            improved = True

    if best_seg is None:
        return None

    return SubsetResult(dims=tuple(selected), segment=best_seg, feature_count=len(selected))


# ----------------------------- Genetic Algorithm --------------------------- #

class GA:
    def __init__(
        self,
        evaluator: SubsetEvaluator,
        dims_pool: Sequence[str],
        population_size: int = 80,
        generations: int = 120,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.05,
        tournament_k: int = 3,
        elitism: int = 2,
        random_seed: Optional[int] = 42,
    ) -> None:
        self.eval = evaluator
        self.dims_pool = list(dims_pool)
        self.M = len(self.dims_pool)
        self.N = int(population_size)
        self.G = int(generations)
        self.cr = float(crossover_rate)
        self.mr = float(mutation_rate)
        self.tk = int(tournament_k)
        self.elitism = int(elitism)
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

    # --- Representation helpers ---
    def random_individual(self) -> np.ndarray:
        # Allow empty subset, but bias toward a few features initially
        probs = np.full(self.M, 0.3)
        chrom = (np.random.rand(self.M) < probs).astype(np.uint8)
        return chrom

    def chrom_to_dims(self, chrom: np.ndarray) -> Tuple[str, ...]:
        return tuple([d for bit, d in zip(chrom, self.dims_pool) if bit == 1])

    # --- Fitness (maximize) ---
    def fitness(self, chrom: np.ndarray) -> Tuple[float, Optional[SegmentMetrics]]:
        dims = self.chrom_to_dims(chrom)
        seg = self.eval.best_segment_for_dims(dims)
        if seg is None:
            return (-1e18, None)  # very bad
        return (seg.score, seg)

    # --- GA operators ---
    def tournament_select(self, pop: List[np.ndarray], fits: List[float]) -> np.ndarray:
        idxs = np.random.choice(len(pop), size=self.tk, replace=False)
        best_idx = idxs[np.argmax([fits[i] for i in idxs])]
        return pop[best_idx].copy()

    def crossover(self, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.rand() > self.cr or self.M <= 1:
            return a.copy(), b.copy()
        # Uniform crossover
        mask = np.random.rand(self.M) < 0.5
        c1 = np.where(mask, a, b).astype(np.uint8)
        c2 = np.where(mask, b, a).astype(np.uint8)
        return c1, c2

    def mutate(self, a: np.ndarray) -> np.ndarray:
        if self.mr <= 0:
            return a
        flips = np.random.rand(self.M) < self.mr
        a = a.copy()
        a[flips] = 1 - a[flips]
        return a

    # --- Main loop ---
    def run(self, top_k: int = 10) -> Tuple[SubsetResult, List[SubsetResult]]:
        # Initialize population
        population = [self.random_individual() for _ in range(self.N)]

        # Evaluate
        fits: List[float] = []
        segs: List[Optional[SegmentMetrics]] = []
        for c in population:
            f, s = self.fitness(c)
            fits.append(f)
            segs.append(s)

        for gen in range(self.G):
            # Elites
            elite_idxs = list(np.argsort(fits)[-self.elitism :])
            elites = [population[i].copy() for i in elite_idxs]
            elite_segs = [segs[i] for i in elite_idxs]
            elite_fits = [fits[i] for i in elite_idxs]

            # New population
            new_pop: List[np.ndarray] = elites.copy()
            while len(new_pop) < self.N:
                p1 = self.tournament_select(population, fits)
                p2 = self.tournament_select(population, fits)
                c1, c2 = self.crossover(p1, p2)
                c1 = self.mutate(c1)
                if len(new_pop) < self.N:
                    new_pop.append(c1)
                if len(new_pop) < self.N:
                    c2 = self.mutate(c2)
                    new_pop.append(c2)

            population = new_pop

            # Evaluate
            fits = []
            segs = []
            for c in population:
                f, s = self.fitness(c)
                fits.append(f)
                segs.append(s)

            # Optional: print progress every 10 generations
            if (gen + 1) % max(1, self.G // 6) == 0:
                best_idx = int(np.argmax(fits))
                best_seg = segs[best_idx]
                best_dims = self.chrom_to_dims(population[best_idx])
                if best_seg is not None:
                    print(
                        f"Gen {gen+1:>3}/{self.G}: best score={fits[best_idx]:.4f} | dims={best_dims} | "
                        f"FTDs={best_seg.ftds} ILS_l2ftd={best_seg.ils_l2ftd:.4f} LSC={best_seg.ils_count}"
                    )

        # Final best
        best_idx = int(np.argmax(fits))
        best_seg = segs[best_idx]
        best_dims = self.chrom_to_dims(population[best_idx])
        assert best_seg is not None
        best = SubsetResult(dims=best_dims, segment=best_seg, feature_count=len(best_dims))

        # Top-K results
        order = np.argsort(fits)[::-1][:top_k]
        top: List[SubsetResult] = []
        for i in order:
            seg = segs[i]
            if seg is None:
                continue
            dims = self.chrom_to_dims(population[i])
            top.append(SubsetResult(dims=dims, segment=seg, feature_count=len(dims)))

        return best, top


# ----------------------------- Data Loading & Prep -------------------------- #

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)

    # Ensure metric columns exist
    missing = [c for c in METRIC_COLS.values() if c not in df.columns]
    if missing:
        raise ValueError(f"Missing metric columns in data: {missing}")

    # Derive ils_l2ftd if not present (we will recompute anyway when grouping)
    if "ils_l2ftd" not in df.columns:
        df["ils_l2ftd"] = df[METRIC_COLS["ftds"]] / df[METRIC_COLS["ils_count"]].replace({0: np.nan})

    # Coerce object-like dimension columns to string and fill NAs
    for col in ALL_DIMENSIONS:
        if col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # keep numeric, but fill NAs consistently for grouping
                df[col] = df[col].fillna(-9999)
            else:
                df[col] = df[col].astype("string").fillna("")
        else:
            # Create missing dimension as blank if absent
            df[col] = ""

    return df


def select_usable_dimensions(df: pd.DataFrame) -> List[str]:
    usable = []
    for d in ALL_DIMENSIONS:
        if d not in df.columns:
            continue

        # If the column is constant overall, skip early.
        nunique_total = df[d].nunique(dropna=False)
        if nunique_total <= 1:
            continue

        # Require at least one non-empty/valid category.
        if pd.api.types.is_string_dtype(df[d]) or isinstance(df[d].dtype, pd.StringDtype):
            nunique_valid = df.loc[df[d] != "", d].nunique(dropna=False)
        elif pd.api.types.is_numeric_dtype(df[d]):
            nunique_valid = df.loc[df[d] != -9999, d].nunique(dropna=False)
        else:
            nunique_valid = df[d].nunique(dropna=False)

        if nunique_valid <= 0:
            continue

        usable.append(d)
    return usable


# ----------------------------- Pretty Printing ----------------------------- #

def print_subset_result(title: str, res: Optional[SubsetResult]) -> None:
    print("\n" + title)
    print("=" * len(title))
    if res is None:
        print("No valid segment found under the given constraints.")
        return
    seg = res.segment
    dims = res.dims
    print(f"Dims ({len(dims)}): {dims}")
    if len(dims) == 0:
        print("Key: [ALL]")
    else:
        key_pairs = ", ".join([f"{d}={v}" for d, v in zip(dims, seg.key)])
        print(f"Key: {key_pairs}")
    print(
        f"FTDs={seg.ftds:,} | LeadStatusCount={seg.ils_count:,} | Accounts={seg.accounts:,} | "
        f"ILS_L2FTD={seg.ils_l2ftd:.6f} | Score={seg.score:.4f}"
    )


def print_top_list(title: str, results: List[SubsetResult], max_rows: int = 10) -> None:
    print("\n" + title)
    print("=" * len(title))
    for i, r in enumerate(results[:max_rows], start=1):
        seg = r.segment
        dims = r.dims
        key_pairs = ", ".join([f"{d}={v}" for d, v in zip(dims, seg.key)]) if dims else "[ALL]"
        print(
            f"{i:>2}. k={len(dims):>2} | dims={dims} | key={key_pairs} | "
            f"FTDs={seg.ftds:,} | LSC={seg.ils_count:,} | ILS_L2FTD={seg.ils_l2ftd:.6f} | Score={seg.score:.4f}"
        )


# ----------------------------- Main ---------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Feature subset optimizer (GA + Greedy)")
    ap.add_argument("--data", type=str, required=True, help="Path to parquet (e.g., datasources/group.parquet)")

    # Objective weights
    ap.add_argument("--w1", type=float, default=1.0, help="Weight for log(1+FTDs)")
    ap.add_argument("--w2", type=float, default=1.0, help="Weight for log(ILS_L2FTD)")

    # Minimum supports
    ap.add_argument("--min-ftds", type=int, default=1, help="Minimum FTDs per segment")
    ap.add_argument(
        "--min-ls", dest="min_ls", type=int, default=1, help="Minimum ils_count per segment"
    )
    ap.add_argument("--min-accounts", type=int, default=1, help="Minimum accounts per segment")

    # Size penalty
    ap.add_argument(
        "--penalty-per-feature",
        type=float,
        default=0.0,
        help="Penalty subtracted from score per included feature (encourages smaller subsets)",
    )

    # GA params
    ap.add_argument("--generations", type=int, default=120)
    ap.add_argument("--population", type=int, default=80)
    ap.add_argument("--crossover", type=float, default=0.8)
    ap.add_argument("--mutation", type=float, default=0.05)
    ap.add_argument("--tournament-k", type=int, default=3)
    ap.add_argument("--elitism", type=int, default=2)
    ap.add_argument("--random-seed", type=int, default=42)

    # Reporting
    ap.add_argument("--top-n", type=int, default=10)

    args = ap.parse_args()

    df = load_data(args.data)
    usable_dims = select_usable_dimensions(df)

    print("Data loaded:")
    print(f" - Rows: {len(df):,}")
    print(f" - Usable dimensions ({len(usable_dims)}): {usable_dims}")

    evaluator = SubsetEvaluator(
        df=df,
        usable_dims=usable_dims,
        w1=args.w1,
        w2=args.w2,
        min_ftds=args.min_ftds,
        min_ils_count=args.min_ls,
        min_accounts=args.min_accounts,
        penalty_per_feature=args.penalty_per_feature,
    )

    # Greedy baseline
    greedy_res = greedy_forward(evaluator, usable_dims)
    print_subset_result("Greedy Forward Best", greedy_res)

    # Genetic Algorithm search
    ga = GA(
        evaluator=evaluator,
        dims_pool=usable_dims,
        population_size=args.population,
        generations=args.generations,
        crossover_rate=args.crossover,
        mutation_rate=args.mutation,
        tournament_k=args.tournament_k,
        elitism=args.elitism,
        random_seed=args.random_seed,
    )

    best, top = ga.run(top_k=args.top_n)

    print_subset_result("GA Best", best)
    print_top_list("GA Top Candidates", top, max_rows=args.top_n)


if __name__ == "__main__":
    main()
