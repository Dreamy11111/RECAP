import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
OUT = RESULTS / "fig_minimal_rollout_comparison.png"


def simulate(method: str, steps: int = 60, switch_step: int = 22, seed: int = 7):
    rng = np.random.default_rng(seed)
    t = np.arange(steps)

    mean_aoi = np.zeros(steps)
    residual = np.zeros(steps)
    recovery = np.zeros(steps)
    mission = np.zeros(steps)

    mean_aoi[0] = 0.85
    residual[0] = 1.05
    mission[0] = 0.18

    if method == "full":
        aoi_decay = 0.90
        residual_decay = 0.88
        mission_gain_pre = 0.018
        mission_gain_post = 0.050
        recovery_sensitivity = 1.45
        semantic_response = 0.16
    else:
        aoi_decay = 0.935
        residual_decay = 0.925
        mission_gain_pre = 0.015
        mission_gain_post = 0.032
        recovery_sensitivity = 1.05
        semantic_response = 0.085

    for k in range(1, steps):
        event_boost = 0.0
        if k >= switch_step:
            event_boost = 0.12 * math.exp(-(k - switch_step) / 10.0)

        link_stress = 0.11 + 0.06 * np.sin(0.17 * k) ** 2 + 0.025 * rng.normal()
        link_stress = max(link_stress, 0.01)

        mean_aoi[k] = max(0.35, aoi_decay * mean_aoi[k - 1] + link_stress)
        if k >= switch_step:
            mean_aoi[k] += 0.08 if method == "rule" else 0.04

        recovery_load = recovery_sensitivity * max(mean_aoi[k] - 1.0, 0.0)
        if k >= switch_step:
            recovery_load += 0.55 if method == "rule" else 0.35
        recovery[k] = 0.72 * recovery[k - 1] + recovery_load

        residual_noise = 0.015 * rng.normal()
        residual[k] = max(
            0.03,
            residual_decay * residual[k - 1]
            + 0.06 * mean_aoi[k]
            + 0.018 * recovery[k]
            + event_boost
            - semantic_response
            + residual_noise,
        )

        mission_gain = mission_gain_post if k >= switch_step else mission_gain_pre
        mission_penalty = 0.020 * residual[k] + 0.008 * mean_aoi[k]
        if k == switch_step:
            mission[k - 1] = max(0.0, mission[k - 1] - (0.05 if method == "full" else 0.09))
        mission[k] = np.clip(
            mission[k - 1] + mission_gain + semantic_response * 0.03 - mission_penalty,
            0.0,
            1.0,
        )

    return {
        "t": t,
        "mission": mission,
        "residual": residual,
        "aoi": mean_aoi,
        "recovery": recovery,
        "switch_step": switch_step,
    }


def main():
    full = simulate("full")
    rule = simulate("rule")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    colors = {"full": "#1f77b4", "rule": "#d62728"}

    for ax in axes.flat:
        ax.axvline(full["switch_step"], color="0.35", linestyle="--", linewidth=1.2)
        ax.text(full["switch_step"] + 0.6, 0.96, "Mission switch", transform=ax.get_xaxis_transform(),
                fontsize=8, color="0.3", va="top")

    axes[0, 0].plot(full["t"], full["mission"], label="Full Hybrid", color=colors["full"], linewidth=2.3)
    axes[0, 0].plot(rule["t"], rule["mission"], label="Planned Rule-Based Semantic", color=colors["rule"], linewidth=2.1)
    axes[0, 0].set_title("(a) Mission response")
    axes[0, 0].set_ylabel("Mission progress")
    axes[0, 0].set_ylim(0, 1.02)
    axes[0, 0].legend(frameon=True, fontsize=8, loc="lower right")

    axes[0, 1].plot(full["t"], full["residual"], color=colors["full"], linewidth=2.3)
    axes[0, 1].plot(rule["t"], rule["residual"], color=colors["rule"], linewidth=2.1)
    axes[0, 1].set_title("(b) Disagreement residual")
    axes[0, 1].set_ylabel("Residual")

    axes[1, 0].plot(full["t"], full["aoi"], color=colors["full"], linewidth=2.3)
    axes[1, 0].plot(rule["t"], rule["aoi"], color=colors["rule"], linewidth=2.1)
    axes[1, 0].set_title("(c) Mean information age")
    axes[1, 0].set_ylabel("AoI (slots)")
    axes[1, 0].set_xlabel("Rollout step")

    axes[1, 1].plot(full["t"], full["recovery"], color=colors["full"], linewidth=2.3)
    axes[1, 1].plot(rule["t"], rule["recovery"], color=colors["rule"], linewidth=2.1)
    axes[1, 1].set_title("(d) Recovery-trigger count")
    axes[1, 1].set_ylabel("Cumulative triggers")
    axes[1, 1].set_xlabel("Rollout step")

    for ax in axes.flat:
        ax.set_xlim(0, full["t"][-1])
        ax.tick_params(labelsize=8)

    fig.suptitle("Minimal synthetic closed-loop comparison", fontsize=13)
    fig.savefig(OUT, dpi=220, bbox_inches="tight")
    print(f"saved {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
