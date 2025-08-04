#!/usr/bin/env python
"""
Monte‑Carlo season‑long WR projections (2025) — winsor + truncated draws
Author : Chris Esposito
Date   : 2025‑08‑04
"""

from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from scipy.stats import truncnorm                       # NEW
import nfl_data_py as nfl

# --------------------------------------------------------------------- #
# 0. configuration
# --------------------------------------------------------------------- #
RND   = np.random.default_rng(42)        # deterministic simulations
N_SIM = 10_000
TARGET_SEASON = 2025

STATS = ["receiving_yards", "receiving_tds", "receptions",
         "rushing_yards",  "rushing_tds",   "fumbles"]

COUNTING = ["receiving_yards", "receiving_tds", "receptions",
            "rushing_yards",  "rushing_tds",
            "rushing_fumbles", "receiving_fumbles", "sack_fumbles"]

THRESH = {"receiving_yards": 200,  "receiving_tds": 2,  "receptions": 20,
          "rushing_yards":   50,   "rushing_tds":   1,  "fumbles":    0}

# --------------------------------------------------------------------- #
# 1. load + clean data
# --------------------------------------------------------------------- #
seasons   = list(range(2013, TARGET_SEASON))          # 2013‑2024
seasonal  = nfl.import_seasonal_data(seasons)
players   = nfl.import_players()

# --- basic cleaning --------------------------------------------------- #
seasonal[COUNTING] = seasonal[COUNTING].fillna(0)

# primary key harmonisation
if "player_id" not in players.columns and "gsis_id" in players.columns:
    players = players.rename(columns={"gsis_id": "player_id"})

# unified fumbles
seasonal["fumbles"] = (
    seasonal[["rushing_fumbles", "receiving_fumbles", "sack_fumbles"]]
    .sum(axis=1)
)

needed_cols = ["player_id", "season"] + STATS
seasonal = seasonal.loc[:, needed_cols]

# --------------------------------------------------------------------- #
# 2. build roster of *currently relevant* WRs
#    - WR position
#    - last_season == 2025 (keeps rookies)
#    - has espn_id  (filters out legacy/inactive ids)
# --------------------------------------------------------------------- #
wr_active = players.query(
    "position == 'WR' and last_season == @TARGET_SEASON and espn_id.notna()"
).copy()

if wr_active.empty:
    raise RuntimeError("WR roster filter returned zero rows. Check players file.")

wr_ids = wr_active.player_id.tolist()

# --------------------------------------------------------------------- #
# 3. YoY log‑ratio distributions  (winsorised)
# --------------------------------------------------------------------- #
wr_seasons = seasonal[seasonal.player_id.isin(wr_ids)].copy()
wr_seasons.sort_values(["player_id", "season"], inplace=True)

dist: dict[str, dict[str, float]] = {}
for stat in STATS:
    prev = wr_seasons.groupby("player_id")[stat].shift(1)
    nxt  = wr_seasons[stat]

    mask = (prev >= THRESH[stat]) & nxt.notna()
    if mask.sum() == 0:
        raise RuntimeError(f"No rows left for {stat} after threshold filter.")

    log_ratio = np.log((nxt[mask] + 1) / (prev[mask] + 1))

    # --- WINORISATION -------------------------------------------------- #
    lower, upper = log_ratio.quantile([0.02, 0.98])
    winsor = log_ratio.clip(lower, upper)

    dist[stat] = {"mu": winsor.mean(), "sigma": winsor.std(ddof=0)}

# --------------------------------------------------------------------- #
# 4. rookie baselines by draft tier
# --------------------------------------------------------------------- #
first_seasons = (
    seasonal.sort_values(["player_id", "season"])
            .drop_duplicates("player_id", keep="first")
)

first_seasons = first_seasons.merge(
    players[["player_id", "draft_round"]],
    on="player_id", how="left"
)

def tier(row) -> str:
    rnd = row["draft_round"]
    return "early" if rnd <= 2 else "mid" if rnd and rnd <= 5 else "late"

first_seasons["tier"] = first_seasons.apply(tier, axis=1)

rookie_means = (
    first_seasons.groupby("tier")[STATS]
                 .mean()
)

# --------------------------------------------------------------------- #
# 5. baseline stats from 2024 (lag‑1)
# --------------------------------------------------------------------- #
baselines_24 = (
    seasonal.query("season == @TARGET_SEASON - 1")
            .set_index("player_id")[STATS]
)

def get_baseline(pid: str, ply_row) -> np.ndarray:
    if pid in baselines_24.index:
        return baselines_24.loc[pid].values
    r = tier(ply_row)
    return rookie_means.loc[r].values

# --------------------------------------------------------------------- #
# 6. helper: truncated‑normal draws  (μ ± 1.5 σ)
# --------------------------------------------------------------------- #
def draw_ratios(mu: float, sigma: float, n: int) -> np.ndarray:
    if sigma == 0:
        return np.full(n, mu)
    return truncnorm.rvs(
        a = -1.5, b = 1.5,         # bounds in standard‑normal space
        loc = mu, scale = sigma,
        size = n,
        random_state = RND
    )

# --------------------------------------------------------------------- #
# 7. Monte‑Carlo simulation
# --------------------------------------------------------------------- #
sim_rows: list[tuple[str, str, np.ndarray]] = []
for _, ply in wr_active.iterrows():
    base = get_baseline(ply.player_id, ply)
    sims = np.zeros((N_SIM, len(STATS)))

    for j, stat in enumerate(STATS):
        mu, sig = dist[stat]["mu"], dist[stat]["sigma"]
        ratios  = draw_ratios(mu, sig, N_SIM)                           # NEW
        sims[:, j] = np.clip((base[j] + 1) * np.exp(ratios) - 1, 0, None)

    sim_rows.append((ply.player_id, ply.display_name, sims))

# --------------------------------------------------------------------- #
# 8. summarise → CSV
# --------------------------------------------------------------------- #
records: list[dict[str, float]] = []
for pid, name, sims in sim_rows:
    df = pd.DataFrame(sims, columns=STATS)
    record = {
        "player_id": pid,
        "display_name": name,
        **{f"mean_{s}":   df[s].mean()        for s in STATS},
        **{f"median_{s}": df[s].median()      for s in STATS},
        **{f"p10_{s}":    df[s].quantile(0.1) for s in STATS},
        **{f"p90_{s}":    df[s].quantile(0.9) for s in STATS},
    }
    records.append(record)

proj = (
    pd.DataFrame(records)
      .round(1)
      .sort_values("mean_receiving_yards", ascending=False)
)

out = Path("wr_sim_projections_2025.csv")
proj.to_csv(out, index=False)
print(f"\nSaved {len(proj):,} WR projections → {out.resolve()}")
print(proj.head(12))
