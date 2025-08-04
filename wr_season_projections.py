#!/usr/bin/env python
"""
Weekly Monte‑Carlo WR projections (2025)

 – AR(1) w/ winsorised residuals + truncnorm
 – Per‑game baseline shrunk toward league mean
 – Beta‑Binomial games‑played modelling
 – Symmetric thresholding + floor on weekly multipliers   <-- NEW
"""

from __future__ import annotations
import warnings, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import truncnorm
import nfl_data_py as nfl

# ------------------------------------------------------------------ #
# Silence common warnings
# ------------------------------------------------------------------ #
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
np.seterr(divide="ignore", invalid="ignore")

# ------------------------------------------------------------------ #
# Configuration
# ------------------------------------------------------------------ #
RND   = np.random.default_rng(42)
N_SIM = 10_000
TARGET_SEASON = 2025
YEARS = list(range(2013, TARGET_SEASON))          # 2013‑2024 history

STATS = ["receiving_yards", "receiving_tds", "receptions"]

THRESH_WK = {"receiving_yards": 15,      # symmetrical filter (prev & next)  <-- RAISED
             "receiving_tds":   1,
             "receptions":      3}

SHRINK_W   = 0.30      # per‑game baseline shrinkage
WIN_LO, WIN_HI = 0.02, 0.98   # winsorisation bounds
LOG_FLOOR  = np.log(0.10)     # weekly drop cannot exceed 90 %             <-- NEW

# ------------------------------------------------------------------ #
# 1. Load data
# ------------------------------------------------------------------ #
week_df  = nfl.import_weekly_data(YEARS)
players  = nfl.import_players()

week_df[STATS] = week_df[STATS].fillna(0)

if "player_id" not in players.columns:
    players = players.rename(columns={"gsis_id": "player_id"})

wr_active = players.query(
    "position=='WR' and last_season==@TARGET_SEASON and espn_id.notna()"
).copy()

wr_ids    = wr_active.player_id.tolist()
wr_weeks  = week_df[week_df.player_id.isin(wr_ids)].copy()
wr_weeks.sort_values(["player_id", "season", "week"], inplace=True)

# ------------------------------------------------------------------ #
# 2. Fit AR(1) on log‑ratios (symmetrically filtered)                <-- UPDATED
# ------------------------------------------------------------------ #
ar_params: dict[str, dict[str, float]] = {}

for stat in STATS:
    grp  = wr_weeks.groupby("player_id")[stat]
    prev = grp.shift(1)
    nxt  = wr_weeks[stat]

    # demand both weeks clear the floor
    mask = (prev >= THRESH_WK[stat]) & (nxt >= THRESH_WK[stat])
    if mask.sum() == 0:
        ar_params[stat] = {"alpha": 0.0, "phi": 0.0, "sigma": 0.0}
        continue

    r = np.log((nxt[mask] + 1) / (prev[mask] + 1))

    r_lag = r.groupby(wr_weeks.loc[mask, "player_id"]).shift(1)
    df = pd.DataFrame({"r": r, "r_lag": r_lag}).dropna()

    if df.empty:
        alpha, phi, resid = r.mean(), 0.0, r - r.mean()
    else:
        var_lag = df["r_lag"].var()
        phi_raw = 0.0 if var_lag == 0 else np.cov(df["r"], df["r_lag"])[0, 1] / var_lag
        phi     = phi_raw if abs(phi_raw) < 0.98 else 0.0
        alpha   = df["r"].mean() - phi * df["r_lag"].mean()
        resid   = df["r"] - alpha - phi * df["r_lag"]

    lo, hi = resid.quantile([WIN_LO, WIN_HI])
    resid  = resid.clip(lo, hi)
    sigma  = resid.std(ddof=0) if np.isfinite(resid.std()) else 0.0

    ar_params[stat] = {"alpha": float(alpha),
                       "phi":   float(phi),
                       "sigma": float(sigma)}

# ------------------------------------------------------------------ #
# 3. Baselines – shrink per‑game 2024 production toward league mean
# ------------------------------------------------------------------ #
wk24 = week_df.query("season == @TARGET_SEASON - 1")

per_game = wk24.groupby("player_id")[STATS].mean()
league_pg = per_game.mean()

per_game = (1 - SHRINK_W) * per_game + SHRINK_W * league_pg

rookie_week = (
    wr_weeks.sort_values(["player_id", "season", "week"])
            .groupby("player_id")
            .head(17)
            .merge(players[["player_id", "draft_round"]],
                   on="player_id", how="left")
)
rookie_week["tier"] = rookie_week["draft_round"].apply(
    lambda r: "early" if r <= 2 else "mid" if r and r <= 5 else "late")
rookie_means = (rookie_week
                .groupby("tier")[STATS]
                .mean()
                .fillna(0.0))
rookie_means = rookie_means.mul(1 - SHRINK_W).add(league_pg * SHRINK_W)

def baseline(pid: str, ply_row) -> np.ndarray:
    if pid in per_game.index:
        return per_game.loc[pid].values
    tier = ("early" if ply_row.draft_round <= 2
            else "mid" if ply_row.draft_round and ply_row.draft_round <= 5
            else "late")
    return rookie_means.loc[tier].values

# ------------------------------------------------------------------ #
# 4. Games‑played model (Beta‑Binomial)
# ------------------------------------------------------------------ #
gp_history = (
    week_df[week_df.player_id.isin(wr_ids)]
    .groupby(["season", "player_id"])
    .receiving_yards
    .apply(lambda s: (s > 0).sum())
)

m = gp_history.mean() / 17
v = gp_history.var(ddof=0) / 17**2
if v == 0 or np.isnan(v):
    alpha_gp, beta_gp = 1.0, 1.0
else:
    alpha_gp = max(1e-6, ((m * (1 - m)) / v - 1) * m)   # guard lower bound
    beta_gp  = max(1e-6, alpha_gp * (1 / m - 1))

def draw_games_played(size: int) -> np.ndarray:
    p = RND.beta(alpha_gp, beta_gp, size)
    return RND.binomial(17, p)

# ------------------------------------------------------------------ #
# 5. Helper – truncated‑normal residual draws
# ------------------------------------------------------------------ #
def draw_eps(sigma: float, size: int) -> np.ndarray:
    if sigma == 0 or not np.isfinite(sigma):
        return np.zeros(size)
    return truncnorm.rvs(-1.5, 1.5, loc=0, scale=sigma,
                         size=size, random_state=RND)

# ------------------------------------------------------------------ #
# 6. Monte‑Carlo simulation
# ------------------------------------------------------------------ #
sim_rows = []
for _, ply in wr_active.iterrows():
    wk0 = baseline(ply.player_id, ply)          # per‑game starting point
    gp  = draw_games_played(N_SIM)              # number of active weeks
    sims = np.zeros((N_SIM, len(STATS)))

    for j, stat in enumerate(STATS):
        p = ar_params[stat]
        eps = draw_eps(p["sigma"], N_SIM * 17).reshape(N_SIM, 17)

        phi, sig = p["phi"], p["sigma"]
        if abs(phi) < 0.98 and sig > 0:
            sd_stationary = sig / np.sqrt(1 - phi ** 2)
            r_prev = RND.normal(p["alpha"] / (1 - phi), sd_stationary, N_SIM)
        else:
            r_prev = np.full(N_SIM, p["alpha"])

        x_prev = np.full(N_SIM, wk0[j])
        totals = np.zeros(N_SIM)

        for w in range(17):
            r_t = p["alpha"] + phi * r_prev + eps[:, w]
            r_t = np.clip(r_t, LOG_FLOOR, None)          # ---- NEW FLOOR -----

            x_raw = np.clip((x_prev + 1) * np.exp(r_t) - 1, 0, None)
            active = (w < gp)            # bool mask per sim
            x_t = np.where(active, x_raw, 0.0)

            totals += x_t
            x_prev = np.where(active, x_raw, x_prev)
            r_prev = r_t

        sims[:, j] = totals

    sim_rows.append((ply.player_id, ply.display_name, sims))

# ------------------------------------------------------------------ #
# 7. Summarise → CSV
# ------------------------------------------------------------------ #
records = []
for pid, name, sims in sim_rows:
    df = pd.DataFrame(sims, columns=[f"tot_{s}" for s in STATS])
    rec = {
        "player_id": pid,
        "display_name": name,
        **{f"mean_{s}":   df[f"tot_{s}"].mean()        for s in STATS},
        **{f"median_{s}": df[f"tot_{s}"].median()      for s in STATS},
        **{f"p10_{s}":    df[f"tot_{s}"].quantile(0.1) for s in STATS},
        **{f"p90_{s}":    df[f"tot_{s}"].quantile(0.9) for s in STATS},
    }
    records.append(rec)

proj = (pd.DataFrame(records)
        .round(1)
        .sort_values("mean_receiving_yards", ascending=False))

out = Path("wr_weekly_projections_2025.csv")
proj.to_csv(out, index=False)
print(f"\nSaved weekly‑based projections for {len(proj)} WRs → {out.resolve()}")
print(f"WRs ≥ 1 000 y (mean): {(proj.mean_receiving_yards >= 1_000).sum()}")
print(proj.head(10))
