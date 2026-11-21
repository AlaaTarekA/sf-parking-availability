"""Pilot step 2: predict "at least one free bay on arrival" and replay drivers.

Uses data/processed/pilot_block_minutes.csv.gz from src/pilot_prepare.py.

Design (Intermediate Report, Section 3.3):
  * Target: block has >= 1 free bay h minutes after the decision time (h = 5, 10, 15).
  * Split by day: train 1, 2, 4, 5 Feb; calibrate 7 Feb; test 8 Feb 2019.
  * Feature sets: A = time and place; B = A + current state of the block.
  * Baselines: "free now stays free" (persistence) and the historical rate for
    the block and hour (training days only).
  * Models: Logistic Regression, Random Forest, XGBoost; isotonic calibration on
    the calibration day.
  * Replay: drivers heading to a busy block choose where to go; success = the
    chosen block has a free bay when they arrive.

    python src/pilot_train.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "results" / "pilot"
SEED = 245
TRAIN_DAYS, CAL_DAYS, TEST_DAYS = (1, 2, 4, 5), (7,), (8,)
HORIZONS = (5, 10, 15)
STEP = 5                       # decision every 5 minutes
BUSY = 0.85                    # busy block = 85%+ occupied now
WALK_M_PER_MIN = 80.0          # ~1.33 m/s walking speed
REPLAY_RADIUS_M = 400
REPLAY_DRIVERS = 10_000
FAIL_COST_MIN = (3, 6, 10)     # cost of reaching a full block, in walking minutes

FEATURES_A = ["hour", "weekday", "is_weekend", "bays", "hist_rate"]
FEATURES_B = FEATURES_A + ["free_now", "occ_now", "any_free_now", "free_lag5", "free_lag15",
                           "delta5", "min_free_15", "mean_free_15"]


def load_matrices():
    panel = pd.read_csv(PROC / "pilot_block_minutes.csv.gz", parse_dates=["time"])
    free = panel.pivot(index="time", columns="block", values="free")
    bays = panel.pivot(index="time", columns="block", values="bays")
    full = pd.date_range(free.index.min().normalize(), free.index.max(), freq="min")
    keep = full[(full.hour >= 7) & (full.hour < 20) & full.normalize().isin(free.index.normalize().unique())]
    free = free.reindex(keep)
    bays = bays.reindex(keep)
    # Fill short gaps within a day only.
    day = free.index.normalize()
    free = free.groupby(day).ffill(limit=3).groupby(day).bfill(limit=3)
    bays = bays.groupby(day).ffill(limit=3).groupby(day).bfill(limit=3)
    blocks = pd.read_csv(PROC / "pilot_blocks.csv").set_index("block").loc[free.columns]
    return free, bays, blocks


def build_examples(free: pd.DataFrame, bays: pd.DataFrame, h: int) -> pd.DataFrame:
    rows = []
    for day_ts, f_day in free.groupby(free.index.normalize()):
        b_day = bays.loc[f_day.index]
        F, B = f_day.to_numpy(), b_day.to_numpy()
        n = len(f_day)
        for i in range(15, n - h, STEP):
            fut = F[i + h]
            now = F[i]
            ok = ~np.isnan(fut) & ~np.isnan(now) & ~np.isnan(F[i - 15]) & (B[i] > 0)
            if not ok.any():
                continue
            window = F[i - 15:i + 1]
            ts = f_day.index[i]
            frame = pd.DataFrame({
                "time": ts, "block": f_day.columns[ok], "day": ts.day,
                "hour": ts.hour + ts.minute / 60, "weekday": ts.weekday(),
                "is_weekend": int(ts.weekday() >= 5), "bays": B[i][ok],
                "free_now": now[ok], "occ_now": 1 - now[ok] / B[i][ok],
                "any_free_now": (now[ok] >= 1).astype(int),
                "free_lag5": F[i - 5][ok], "free_lag15": F[i - 15][ok],
                "delta5": now[ok] - F[i - 5][ok],
                "min_free_15": np.nanmin(window[:, ok], axis=0),
                "mean_free_15": np.nanmean(window[:, ok], axis=0),
                "free_future": fut[ok],
            })
            rows.append(frame)
    data = pd.concat(rows, ignore_index=True)
    data["y"] = (data["free_future"] >= 1).astype(int)
    for col in ["free_lag5", "free_lag15", "delta5"]:
        data[col] = data[col].fillna(data["free_now"] if col != "delta5" else 0)
    return data


def add_history(data: pd.DataFrame) -> pd.DataFrame:
    train = data[data["day"].isin(TRAIN_DAYS)]
    train_hour = train["hour"].astype(int)
    rate = train.groupby([train["block"], train_hour])["y"].mean()
    block_rate = train.groupby("block")["y"].mean()
    key = pd.MultiIndex.from_arrays([data["block"], data["hour"].astype(int)])
    data["hist_rate"] = rate.reindex(key).to_numpy()
    data["hist_rate"] = data["hist_rate"].fillna(data["block"].map(block_rate)).fillna(train["y"].mean())
    return data


def models():
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=50, n_jobs=-1,
                                                random_state=SEED),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8, eval_metric="logloss", random_state=SEED,
                                 n_jobs=-1),
    }


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def scores(y, p, busy):
    out = {"brier": brier_score_loss(y, p), "brier_busy": brier_score_loss(y[busy], p[busy]),
           "ece": ece(y, p)}
    if len(np.unique(y)) > 1 and len(np.unique(p)) > 1:
        out["auc"] = roc_auc_score(y, p)
    return {k: round(float(v), 4) for k, v in out.items()}


def evaluate(data: pd.DataFrame, h: int):
    train = data[data["day"].isin(TRAIN_DAYS)]
    cal = data[data["day"].isin(CAL_DAYS)]
    test = data[data["day"].isin(TEST_DAYS)]
    y = test["y"].to_numpy()
    busy = (test["occ_now"] >= BUSY).to_numpy()
    rows, fitted = [], {}
    rows.append({"horizon_min": h, "model": "Persistence (free now stays free)", "features": "-",
                 **scores(y, test["any_free_now"].to_numpy().astype(float), busy)})
    rows.append({"horizon_min": h, "model": "Historical rate (block x hour)", "features": "-",
                 **scores(y, test["hist_rate"].to_numpy(), busy)})
    for fs_name, cols in (("A", FEATURES_A), ("B", FEATURES_B)):
        for name, model in models().items():
            model.fit(train[cols], train["y"])
            raw = model.predict_proba(test[cols])[:, 1]
            calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
            calibrated.fit(cal[cols], cal["y"])
            p = calibrated.predict_proba(test[cols])[:, 1]
            s = scores(y, p, busy)
            s["ece_before_calibration"] = round(ece(y, raw), 4)
            rows.append({"horizon_min": h, "model": name, "features": fs_name, **s})
            fitted[(fs_name, name)] = (calibrated, p)
    return pd.DataFrame(rows), fitted, test


def replay(test: pd.DataFrame, prob: np.ndarray, free_future: pd.Series, dist: pd.DataFrame,
           rng: np.random.Generator) -> pd.DataFrame:
    test = test.assign(p=prob).set_index(["time", "block"])
    busy_rows = test[test["occ_now"] >= BUSY].reset_index()
    picks = busy_rows.sample(n=REPLAY_DRIVERS, replace=len(busy_rows) < REPLAY_DRIVERS,
                             random_state=SEED)
    results = []
    by_time = {t: g.droplevel(0) for t, g in test.groupby(level=0)}
    for _, d in picks.iterrows():
        snap = by_time[d["time"]]
        near = dist.loc[d["block"]]
        cand = near[(near <= REPLAY_RADIUS_M) & near.index.isin(snap.index)]
        snap = snap.loc[cand.index]
        walk = cand.to_numpy()
        success = (snap["free_future"].to_numpy() >= 1)
        choice = {"Nearest block (destination)": int(np.argmin(walk))}
        free_now = snap["any_free_now"].to_numpy() == 1
        choice["Nearest block free now"] = (int(np.argmin(np.where(free_now, walk, np.inf)))
                                            if free_now.any() else choice["Nearest block (destination)"])
        for c in FAIL_COST_MIN:
            cost = walk + (1 - snap["p"].to_numpy()) * c * WALK_M_PER_MIN
            choice[f"Model recommendation (fail cost {c} min)"] = int(np.argmin(cost))
        for name, k in choice.items():
            results.append({"strategy": name, "success": bool(success[k]), "walk_m": float(walk[k])})
    res = pd.DataFrame(results)
    return (res.groupby("strategy").agg(first_try_success=("success", "mean"), avg_walk_m=("walk_m", "mean"))
            .round({"first_try_success": 3, "avg_walk_m": 1}).sort_values("first_try_success")
            .reset_index())


def distance_table(blocks: pd.DataFrame) -> pd.DataFrame:
    lat, lon = np.radians(blocks["lat"].to_numpy()), np.radians(blocks["lon"].to_numpy())
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    d = 2 * 6_371_000 * np.arcsin(np.sqrt(a))
    return pd.DataFrame(d, index=blocks.index, columns=blocks.index)


def figures(metrics: pd.DataFrame, rep: pd.DataFrame, cal_curve: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir = ROOT / "results" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    m = metrics[(metrics["features"].isin(["B", "-"]))]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for ax, col, title in ((axes[0], "brier", "All blocks"), (axes[1], "brier_busy", "Busy blocks (85%+ occupied)")):
        for name, g in m.groupby("model"):
            ax.plot(g["horizon_min"], g[col], marker="o", label=name)
        ax.set_title(f"Test Brier score, {title}"); ax.set_xlabel("Minutes until arrival")
        ax.set_ylabel("Brier score (lower is better)"); ax.set_xticks(HORIZONS)
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figdir / "pilot_brier_by_horizon.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="#999")
    for label, (x, y) in cal_curve.items():
        ax.plot(x, y, marker="o", label=label)
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed share with a free bay")
    ax.set_title("Calibration, Random Forest (10 min)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figdir / "pilot_calibration.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.8))
    colors = ["#1F8A80" if s.startswith("Model") else "#9AA5A8" for s in rep["strategy"]]
    ax.barh(rep["strategy"], rep["first_try_success"] * 100, color=colors)
    for i, r in enumerate(rep.itertuples()):
        ax.text(r.first_try_success * 100 + 0.8, i, f"{r.first_try_success:.0%} · walk {r.avg_walk_m:.0f} m",
                va="center", fontsize=9)
    ax.set_xlim(0, 110); ax.set_xlabel("Found a space on the first block tried (%)")
    ax.set_title("Replay: 10,000 drivers heading to a busy block, arriving in 10 min")
    fig.tight_layout(); fig.savefig(figdir / "pilot_replay.png", dpi=160); plt.close(fig)


def calibration_points(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    xs, ys = [], []
    for b in range(bins):
        m = idx == b
        if m.sum() >= 30:
            xs.append(p[m].mean()); ys.append(y[m].mean())
    return xs, ys


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    free, bays, blocks = load_matrices()
    dist = distance_table(blocks)
    all_metrics, summary = [], {}
    rng = np.random.default_rng(SEED)
    for h in HORIZONS:
        data = add_history(build_examples(free, bays, h))
        metrics, fitted, test = evaluate(data, h)
        all_metrics.append(metrics)
        summary[f"rows_{h}min"] = {k: int(data["day"].isin(v).sum()) for k, v in
                                   (("train", TRAIN_DAYS), ("calibration", CAL_DAYS), ("test", TEST_DAYS))}
        if h == 10:
            model, p = fitted[("B", "Random Forest")]
            rf = models()["Random Forest"].fit(data[data["day"].isin(TRAIN_DAYS)][FEATURES_B],
                                               data[data["day"].isin(TRAIN_DAYS)]["y"])
            raw = rf.predict_proba(test[FEATURES_B])[:, 1]
            y = test["y"].to_numpy()
            cal_curve = {"Before calibration": calibration_points(y, raw),
                         "After isotonic calibration": calibration_points(y, p)}
            rep = replay(test, p, test["free_future"], dist, rng)
            rep.to_csv(OUT / "replay_10min.csv", index=False)
    metrics = pd.concat(all_metrics, ignore_index=True)
    metrics.to_csv(OUT / "test_metrics.csv", index=False)

    best = {}
    for h in HORIZONS:
        m = metrics[metrics["horizon_min"] == h].set_index(["model", "features"])
        pers = m.loc[("Persistence (free now stays free)", "-")]
        rf = m.loc[("Random Forest", "B")]
        best[f"{h}min"] = {
            "rf_brier": rf["brier"], "persistence_brier": pers["brier"],
            "rf_brier_reduction_vs_persistence": round(1 - rf["brier"] / pers["brier"], 3),
            "rf_brier_reduction_busy": round(1 - rf["brier_busy"] / pers["brier_busy"], 3),
            "rf_ece_before": rf["ece_before_calibration"], "rf_ece_after": rf["ece"],
        }
    summary.update({"blocks": int(free.shape[1]), "decision_step_min": STEP,
                    "random_forest_vs_persistence": best, "runtime_seconds": round(time.time() - t0, 1)})
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    figures(metrics, rep, cal_curve)
    print(metrics.to_string(index=False))
    print(rep.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
