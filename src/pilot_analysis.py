"""Pilot step 3: extra analysis for the final report (10-minute horizon).

* Data overview: occupancy and share of full blocks by hour; spread of block occupancy.
* Permutation feature importance of the calibrated Random Forest (feature set B).
* Brier score by how busy the block is at decision time, model vs persistence.
* Precision-recall curve for the "full on arrival" class.
* Replay sensitivity: walking radius 200 m vs 400 m.

    python src/pilot_analysis.py      (after src/pilot_train.py's data exists)
Outputs: results/pilot/analysis/*.csv, results/figures/pilot_*.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, precision_recall_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pilot_train as pt  # noqa: E402

OUT = pt.ROOT / "results" / "pilot" / "analysis"
FIG = pt.ROOT / "results" / "figures"
H = 10
LABELS = {"hour": "Hour of day", "weekday": "Day of week", "is_weekend": "Weekend",
          "bays": "Spaces on block", "hist_rate": "Historical free rate",
          "free_now": "Free spaces now", "occ_now": "Occupancy now", "any_free_now": "Any space free now",
          "free_lag5": "Free spaces 5 min ago", "free_lag15": "Free spaces 15 min ago",
          "delta5": "Change in last 5 min", "min_free_15": "Min free, last 15 min",
          "mean_free_15": "Mean free, last 15 min"}


def plt_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt = plt_setup()
    free, bays, blocks = pt.load_matrices()
    occ = 1 - free / bays

    # 1. Data overview
    by_hour = pd.DataFrame({
        "mean_occupancy": occ.groupby(occ.index.hour).mean().mean(axis=1),
        "share_block_minutes_full": (free <= 0).where(free.notna()).groupby(free.index.hour).mean().mean(axis=1),
    })
    by_hour.index.name = "hour"
    by_hour.round(4).to_csv(OUT / "occupancy_by_hour.csv")
    block_mean = occ.mean()
    spread = {"blocks": int(occ.shape[1]), "mean_occupancy": round(float(occ.stack().mean()), 3),
              "block_mean_occupancy_quartiles": [round(float(q), 3) for q in block_mean.quantile([.25, .5, .75])],
              "blocks_over_85pct_on_average": int((block_mean >= .85).sum()),
              "share_block_minutes_full": round(float((free <= 0).where(free.notna()).stack().mean()), 4)}

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].plot(by_hour.index, by_hour.mean_occupancy * 100, marker="o", color="#1F8A80", label="Mean occupancy")
    ax2 = axes[0].twinx()
    ax2.bar(by_hour.index, by_hour.share_block_minutes_full * 100, alpha=.35, color="#E0A21B",
            label="Block-minutes completely full")
    axes[0].set(xlabel="Hour of day", ylabel="Mean occupancy (%)", title="Occupancy and full blocks by hour")
    ax2.set_ylabel("Completely full (%)")
    axes[0].legend(loc="upper left", fontsize=8); ax2.legend(loc="upper right", fontsize=8)
    axes[1].hist(block_mean * 100, bins=25, color="#1F8A80")
    axes[1].set(xlabel="Average occupancy of block (%)", ylabel="Blocks", title="How busy blocks are on average")
    fig.tight_layout(); fig.savefig(FIG / "pilot_data_overview.png", dpi=150); plt.close(fig)

    # Model at 10 minutes
    data = pt.add_history(pt.build_examples(free, bays, H))
    train, cal, test = (data[data["day"].isin(d)] for d in (pt.TRAIN_DAYS, pt.CAL_DAYS, pt.TEST_DAYS))
    cols = pt.FEATURES_B
    models = {}
    for name in ("Random Forest", "Logistic Regression"):
        m = pt.models()[name].fit(train[cols], train["y"])
        models[name] = CalibratedClassifierCV(FrozenEstimator(m), method="isotonic").fit(cal[cols], cal["y"])
    y = test["y"].to_numpy()
    p_rf = models["Random Forest"].predict_proba(test[cols])[:, 1]
    p_lr = models["Logistic Regression"].predict_proba(test[cols])[:, 1]
    p_pers = test["any_free_now"].to_numpy().astype(float)

    # 2. Permutation importance (Brier), on a fixed sample of the test day
    sample = test.sample(min(12000, len(test)), random_state=pt.SEED)
    imp = permutation_importance(models["Random Forest"], sample[cols], sample["y"],
                                 scoring="neg_brier_score", n_repeats=5, random_state=pt.SEED, n_jobs=-1)
    importance = pd.DataFrame({"feature": cols, "brier_increase": imp.importances_mean,
                               "std": imp.importances_std}).sort_values("brier_increase", ascending=False)
    importance.round(6).to_csv(OUT / "permutation_importance_rf_B_10min.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    top = importance.iloc[::-1]
    ax.barh([LABELS[f] for f in top.feature], top.brier_increase * 1000, xerr=top["std"] * 1000,
            color="#1F8A80")
    ax.set(xlabel="Increase in Brier score when shuffled (x 1000)",
           title="Permutation importance, Random Forest (10 min)")
    fig.tight_layout(); fig.savefig(FIG / "pilot_importance.png", dpi=150); plt.close(fig)

    # 3. Brier by busyness at decision time
    bins = [-0.01, .5, .7, .85, .95, 1.0]
    names = ["under 50%", "50-70%", "70-85%", "85-95%", "95-100%"]
    test = test.assign(p_rf=p_rf, p_pers=p_pers, band=pd.cut(test["occ_now"], bins, labels=names))
    rows = []
    for band, g in test.groupby("band", observed=True):
        rows.append({"occupancy_now": band, "examples": len(g), "share_full_on_arrival": round(1 - g["y"].mean(), 4),
                     "brier_persistence": round(((g.p_pers - g.y) ** 2).mean(), 4),
                     "brier_random_forest": round(((g.p_rf - g.y) ** 2).mean(), 4)})
    bands = pd.DataFrame(rows)
    bands["reduction"] = (1 - bands.brier_random_forest / bands.brier_persistence).round(3)
    bands.to_csv(OUT / "brier_by_occupancy.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(len(bands))
    ax.bar(x - .2, bands.brier_persistence, .4, label="Persistence", color="#9AA5A8")
    ax.bar(x + .2, bands.brier_random_forest, .4, label="Random Forest", color="#1F8A80")
    ax.set_xticks(x, bands.occupancy_now)
    ax.set(xlabel="Block occupancy at decision time", ylabel="Brier score (lower is better)",
           title="Error by how busy the block is (10 min)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "pilot_brier_by_occupancy.png", dpi=150); plt.close(fig)

    # 4. Precision-recall for "full on arrival"
    full = 1 - y
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    pr = {}
    for name, p, col in (("Random Forest", p_rf, "#1F8A80"), ("Logistic Regression", p_lr, "#E0A21B")):
        prec, rec, _ = precision_recall_curve(full, 1 - p)
        ap = average_precision_score(full, 1 - p)
        pr[name] = round(float(ap), 4)
        ax.plot(rec, prec, color=col, label=f"{name} (AP {ap:.2f})")
    tp = ((p_pers == 0) & (full == 1)).sum()
    pers_p, pers_r = tp / max((p_pers == 0).sum(), 1), tp / full.sum()
    ax.scatter([pers_r], [pers_p], color="#33424A", zorder=3, label="Persistence (one point)")
    ax.axhline(full.mean(), ls="--", color="#999", lw=1, label=f"Random guess ({full.mean():.3f})")
    ax.set(xlabel="Recall (full blocks found)", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02),
           title="Precision-recall, full on arrival (10 min)")
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(FIG / "pilot_precision_recall.png", dpi=150); plt.close(fig)

    # 5. Replay sensitivity to walking radius
    rng = np.random.default_rng(pt.SEED)
    dist = pt.distance_table(blocks)
    reps = []
    for radius in (200, 400):
        pt.REPLAY_RADIUS_M = radius
        r = pt.replay(test.drop(columns=["p_rf", "p_pers", "band"]), p_rf, test["free_future"], dist, rng)
        reps.append(r.assign(radius_m=radius))
    pt.REPLAY_RADIUS_M = 400
    replay = pd.concat(reps, ignore_index=True)
    replay.to_csv(OUT / "replay_by_radius.csv", index=False)

    summary = {"data_overview": spread, "average_precision_full": pr,
               "persistence_point": {"precision": round(float(pers_p), 4), "recall": round(float(pers_r), 4)},
               "top_features": importance.head(5)[["feature", "brier_increase"]].round(6).to_dict("records")}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(by_hour.round(3).to_string()); print(json.dumps(summary, indent=2))
    print(importance.round(5).to_string(index=False)); print(bands.to_string(index=False))
    print(replay.to_string(index=False))


if __name__ == "__main__":
    main()
