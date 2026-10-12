# Turn the raw benchmark episodes into README figures + a summary table.
#
# Reads results/benchmarks/{scaling,suite}.jsonl (see run_benchmarks.py) and
# writes:
#   assets/figures/<name>-{light,dark}.png   one figure per theme
#   benchmarks/summary.csv                   per (config, team size) statistics
#
# Statistics:
#   * completion rate with a Wilson 95% interval;
#   * speedup over a single robot, PAIRED per map (same map, the n=1 run of the
#     same configuration), summarised by the median with a bootstrap 95% CI.
#     Time is ticks to 90% coverage; a run that never reaches 90% (deadlock) is
#     charged the tick cap, so failures pull the speedup down rather than being
#     silently dropped;
#   * sensing redundancy (duplicated first-hand sensing) mean with bootstrap CI.

from __future__ import annotations

import csv
import json
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from default_maps import DEFAULT_MAPS

RAW = os.path.join(REPO, "results", "benchmarks")
FIG_DIR = os.path.join(REPO, "assets", "figures")
RNG = np.random.default_rng(0)
N_BOOT = 4000

# Display order == categorical slot order (best-performing first). Colour follows
# the configuration everywhere.
ORDER = ["coordinated_comms", "greedy_comms", "coordinated", "independent_comms", "independent"]
LABELS = {
    "coordinated_comms": "Coordinated (Hungarian) + shared maps",
    "greedy_comms": "Coordinated (greedy) + shared maps",
    "coordinated": "Coordinated (Hungarian), no comms",
    "independent_comms": "Independent + shared maps",
    "independent": "Independent, no comms",
}
SHORT = {
    "coordinated_comms": "Hungarian\n+ comms",
    "greedy_comms": "Greedy\n+ comms",
    "coordinated": "Hungarian\nno comms",
    "independent_comms": "Independent\n+ comms",
    "independent": "Independent\nno comms",
}

THEMES = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"],
        "ramp": ["#f0efec", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
        "free": "#f0efec", "wall": "#52514e",
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"],
        "ramp": ["#383835", "#104281", "#1c5cab", "#2a78d6", "#5598e7", "#9ec5f4"],
        "free": "#383835", "wall": "#c3c2b7",
    },
}


# -- data ------------------------------------------------------------------


def load(name: str) -> list[dict]:
    with open(os.path.join(RAW, f"{name}.jsonl")) as f:
        return [json.loads(line) for line in f]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def boot_ci(values: np.ndarray, stat=np.median) -> tuple[float, float]:
    idx = RNG.integers(0, len(values), size=(N_BOOT, len(values)))
    samples = stat(values[idx], axis=1)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def t90(ep: dict) -> int:
    return ep["ticks_to_90"] if ep["ticks_to_90"] is not None else ep["max_ticks"]


def t100(ep: dict) -> int:
    # Timeout-inclusive time to full coverage: unfinished runs are charged the cap.
    return ep["ticks_to_100"] if ep["completed"] else ep["max_ticks"]


def mcnemar(a: list[bool], b: list[bool]) -> tuple[int, int, float]:
    # Exact two-sided McNemar test on paired completion outcomes.
    only_a = sum(x and not y for x, y in zip(a, b))
    only_b = sum(y and not x for x, y in zip(a, b))
    m = only_a + only_b
    if m == 0:
        return only_a, only_b, 1.0
    tail = sum(math.comb(m, i) for i in range(min(only_a, only_b) + 1)) / 2**m
    return only_a, only_b, min(1.0, 2 * tail)


def summarise(scaling: list[dict]) -> list[dict]:
    by = {}
    for ep in scaling:
        by.setdefault((ep["config"], ep["n"]), {})[ep["root"]] = ep
    rows = []
    for config in ORDER:
        solo = by[(config, 1)]
        for n in (1, 2, 4, 8):
            eps = by[(config, n)]
            roots = sorted(eps)
            done = sum(eps[r]["completed"] for r in roots)
            lo, hi = wilson(done, len(roots))
            speed = np.array([t90(solo[r]) / t90(eps[r]) for r in roots])
            red = np.array([eps[r]["redundancy"] for r in roots])
            capped = np.array([t100(eps[r]) for r in roots])
            speed_ci = boot_ci(speed)
            red_ci = boot_ci(red, np.mean)
            completed_ticks = [eps[r]["ticks"] for r in roots if eps[r]["completed"]]
            rows.append({
                "config": config,
                "label": LABELS[config],
                "robots": n,
                "episodes": len(roots),
                "completion_rate": done / len(roots),
                "completion_ci_lo": lo,
                "completion_ci_hi": hi,
                "deadlock_rate": sum(eps[r]["deadlocked"] for r in roots) / len(roots),
                "median_ticks_to_90": float(np.median([t90(eps[r]) for r in roots])),
                "median_ticks_to_100_completed": (
                    float(np.median(completed_ticks)) if completed_ticks else float("nan")
                ),
                "median_ticks_to_100_capped": float(np.median(capped)),
                "mean_ticks_to_100_capped": float(capped.mean()),
                "mean_final_coverage": float(np.mean([eps[r]["final_coverage"] for r in roots])),
                "median_speedup_t90": float(np.median(speed)),
                "speedup_ci_lo": speed_ci[0],
                "speedup_ci_hi": speed_ci[1],
                "mean_redundancy": float(red.mean()),
                "redundancy_ci_lo": red_ci[0],
                "redundancy_ci_hi": red_ci[1],
                "mean_conflicts": float(np.mean([eps[r]["conflicts"] for r in roots])),
                "mean_payload_kb_delivered": float(
                    np.mean([eps[r]["payload_bytes_delivered"] for r in roots]) / 1024
                ),
                "mean_wire_kb_delivered": float(
                    np.mean([eps[r]["wire_bytes_delivered"] for r in roots]) / 1024
                ),
            })
    return rows


# -- styling ---------------------------------------------------------------


def style_axes(ax, th) -> None:
    ax.set_facecolor(th["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(th["axis"])
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=th["muted"], labelcolor=th["ink2"], labelsize=9, length=0, pad=6)
    ax.grid(True, axis="y", color=th["grid"], linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(th["ink2"])
    ax.yaxis.label.set_color(th["ink2"])


def new_fig(th, w, h, **kw):
    fig, axes = plt.subplots(figsize=(w, h), dpi=200, **kw)
    fig.patch.set_facecolor(th["surface"])
    return fig, axes


def title(fig, th, text, sub) -> None:
    fig.text(0.012, 0.975, text, ha="left", va="top", fontsize=13, fontweight="bold", color=th["ink"])
    fig.text(0.012, 0.915, sub, ha="left", va="top", fontsize=9.5, color=th["ink2"])


def legend(fig, ax_or_handles, th, ncol, y) -> None:
    handles, labels = ax_or_handles.get_legend_handles_labels()
    leg = fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol,
        frameon=False, fontsize=8.5, handlelength=1.6, columnspacing=1.6,
    )
    for text in leg.get_texts():
        text.set_color(th["ink2"])


def save(fig, name, theme) -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, f"{name}-{theme}.png")
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("wrote", os.path.relpath(path, REPO))


# -- figures ---------------------------------------------------------------


def fig_coverage(scaling, th, theme, n=4, horizon=500) -> None:
    fig, ax = new_fig(th, 9, 4.6)
    fig.subplots_adjust(left=0.075, right=0.975, top=0.80, bottom=0.25)
    style_axes(ax, th)
    t = np.arange(horizon + 1)

    def curves(config, team):
        eps = [e for e in scaling if e["config"] == config and e["n"] == team]
        out = np.empty((len(eps), horizon + 1))
        for i, e in enumerate(eps):
            c = np.asarray(e["coverage"][: horizon + 1])
            # A finished or deadlocked team keeps the coverage it reached.
            out[i] = np.concatenate([c, np.full(horizon + 1 - len(c), c[-1])])
        return out * 100

    solo = curves("independent", 1).mean(axis=0)
    ax.plot(t, solo, color=th["muted"], linewidth=1.5, label="1 robot (reference)")
    for k, config in enumerate(ORDER):
        cov = curves(config, n)
        mean = cov.mean(axis=0)
        idx = RNG.integers(0, len(cov), size=(1000, len(cov)))
        boots = cov[idx].mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
        color = th["series"][k]
        ax.fill_between(t, lo, hi, color=color, alpha=0.12, linewidth=0)
        ax.plot(t, mean, color=color, linewidth=2, solid_capstyle="round", label=LABELS[config])

    ax.set_xlim(0, horizon)
    ax.set_ylim(0, 102)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Simulation tick")
    ax.set_ylabel("Map explored by the team")
    title(
        fig, th, f"Classical baselines: coverage over time, {n}-robot teams",
        "Mean over 200 random 25×25 maps (shaded: 95% bootstrap CI). Flat tails are teams stuck in a deadlock.",
    )
    legend(fig, ax, th, ncol=3, y=0.0)
    save(fig, "coverage_curves", theme)


def fig_scaling(rows, th, theme) -> None:
    fig, axes = new_fig(th, 11, 4.4, ncols=3)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.74, bottom=0.25, wspace=0.32)
    ns = [1, 2, 4, 8]
    panels = [
        ("Runs that explore the whole map", "completion_rate", "completion_ci_lo", "completion_ci_hi", 100, "%"),
        ("Timeout-capped speedup to 90% explored", "median_speedup_t90", "speedup_ci_lo", "speedup_ci_hi", 1, "×"),
        ("Duplicated sensing (redundancy)", "mean_redundancy", "redundancy_ci_lo", "redundancy_ci_hi", 1, ""),
    ]
    for ax, (name, key, lo_k, hi_k, scale, unit) in zip(axes, panels):
        style_axes(ax, th)
        if key == "median_speedup_t90":
            ax.plot(ns, ns, color=th["muted"], linewidth=1.5, label="Ideal (linear)")
            ax.text(7.4, 8.1, "ideal (linear)", color=th["muted"], fontsize=8.5, ha="right", va="bottom")
        for k, config in enumerate(ORDER):
            sub = [r for r in rows if r["config"] == config]
            y = np.array([r[key] for r in sub]) * scale
            lo = np.array([r[lo_k] for r in sub]) * scale
            hi = np.array([r[hi_k] for r in sub]) * scale
            color = th["series"][k]
            ax.fill_between(ns, lo, hi, color=color, alpha=0.12, linewidth=0)
            ax.plot(ns, y, color=color, linewidth=2, label=LABELS[config] if ax is axes[0] else None)
            ax.scatter(ns, y, s=34, color=color, edgecolors=th["surface"], linewidths=2, zorder=3)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in ns])
        ax.minorticks_off()
        ax.set_xlabel("Robots on the team")
        ax.set_title(name, loc="left", fontsize=10, color=th["ink"], pad=10)
        if unit == "%":
            ax.set_ylim(0, 105)
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
        elif unit == "×":
            ax.set_ylim(0, 8.8)
            ax.set_yticks([0, 2, 4, 6, 8])
            ax.set_yticklabels(["0×", "2×", "4×", "6×", "8×"])
        else:
            ax.set_ylim(bottom=0)
    title(
        fig, th, "Classical baselines: how each strategy scales with team size",
        "200 random 25×25 maps per point, paired across strategies (same maps, same spawns). Shaded: 95% CI.\n"
        "Speedup is the per-map median; a run that never reaches 90% (deadlock) is charged the 1,500-tick cap.",
    )
    legend(fig, axes[0], th, ncol=3, y=0.0)
    save(fig, "scaling", theme)


def fig_suite(suite, th, theme) -> None:
    names = list(DEFAULT_MAPS)
    mat = np.zeros((len(names), len(ORDER)))
    for i, name in enumerate(names):
        for j, config in enumerate(ORDER):
            eps = [e for e in suite if e["map_name"] == name and e["config"] == config]
            mat[i, j] = 100 * np.mean([e["completed"] for e in eps])
    n_spawns = len([e for e in suite if e["map_name"] == names[0] and e["config"] == ORDER[0]])

    fig = plt.figure(figsize=(9, 9.4), dpi=200)
    fig.patch.set_facecolor(th["surface"])
    gs = fig.add_gridspec(len(names), 2, width_ratios=[0.55, len(ORDER)], left=0.17, right=0.98,
                          top=0.865, bottom=0.03, wspace=0.04, hspace=0.12)
    cmap = LinearSegmentedColormap.from_list("seq", th["ramp"])
    heat = fig.add_subplot(gs[:, 1])
    heat.imshow(mat, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    heat.set_facecolor(th["surface"])
    for s in heat.spines.values():
        s.set_visible(False)
    heat.set_xticks(range(len(ORDER)))
    heat.set_xticklabels([SHORT[c] for c in ORDER], fontsize=8.5, color=th["ink2"])
    heat.xaxis.tick_top()
    heat.set_yticks([])
    heat.tick_params(length=0)
    # 2px surface gaps between cells.
    for x in np.arange(-0.5, len(ORDER), 1):
        heat.axvline(x, color=th["surface"], linewidth=2.5)
    for y in np.arange(-0.5, len(names), 1):
        heat.axhline(y, color=th["surface"], linewidth=2.5)
    for i in range(len(names)):
        for j in range(len(ORDER)):
            v = mat[i, j]
            rgb = np.array(cmap(v / 100)[:3])
            lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
            heat.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=8.5,
                      color="#0b0b0b" if lum > 0.5 else "#ffffff")

    free, wall = th["free"], th["wall"]
    thumb_cmap = LinearSegmentedColormap.from_list("map", [free, wall])
    for i, name in enumerate(names):
        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(DEFAULT_MAPS[name]["grid"], cmap=thumb_cmap, vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.text(-0.6, 0.5, name.replace("_", " "), transform=ax.transAxes, ha="right", va="center",
                fontsize=8.5, color=th["ink2"])

    fig.text(0.012, 0.985, "Classical baselines on the stress-test suite: % of runs that fully explore each map",
             ha="left", va="top", fontsize=13, fontweight="bold", color=th["ink"])
    fig.text(0.012, 0.962,
             f"18 handcrafted 10×10 maps, each targeting a failure mode. 3 robots, {n_spawns} random spawns per cell. "
             "Failures are deadlocks.",
             ha="left", va="top", fontsize=9.5, color=th["ink2"])
    save(fig, "stress_suite", theme)


def write_paired_tests(scaling) -> None:
    # Completion with vs. without map sharing, paired on the same map and spawns.
    by = {(e["config"], e["n"], e["root"]): e["completed"] for e in scaling}
    roots = sorted({e["root"] for e in scaling})
    path = os.path.join(REPO, "benchmarks", "paired_tests.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["comparison", "robots", "maps", "rate_with_comms", "rate_without_comms",
                         "only_with_comms_completed", "only_without_comms_completed", "mcnemar_p"])
        for with_c, without_c in (("coordinated_comms", "coordinated"), ("independent_comms", "independent")):
            for n in (2, 4, 8):
                a = [by[(with_c, n, r)] for r in roots]
                b = [by[(without_c, n, r)] for r in roots]
                only_a, only_b, p = mcnemar(a, b)
                writer.writerow([f"{with_c} vs {without_c}", n, len(roots), round(np.mean(a), 4),
                                 round(np.mean(b), 4), only_a, only_b, f"{p:.3g}"])
                print(f"  {with_c} vs {without_c}, n={n}: {np.mean(a):.3f} vs {np.mean(b):.3f}, "
                      f"discordant {only_a}/{only_b}, McNemar p={p:.3g}")
    print("wrote", os.path.relpath(path, REPO))


def write_summary(rows) -> None:
    path = os.path.join(REPO, "benchmarks", "summary.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    print("wrote", os.path.relpath(path, REPO))


def print_headlines(rows, scaling, suite) -> None:
    print("\nconfig                 n  done%   [CI]          dead%  T90med  T100med  speedup [CI]        redund  conflicts  payloadKB")
    for r in rows:
        print(
            f"{r['config']:22s} {r['robots']}  {r['completion_rate']*100:5.1f} "
            f"[{r['completion_ci_lo']*100:4.1f},{r['completion_ci_hi']*100:5.1f}]  {r['deadlock_rate']*100:5.1f}  "
            f"{r['median_ticks_to_90']:6.0f}  {r['median_ticks_to_100_completed']:7.0f}  "
            f"{r['median_speedup_t90']:4.2f} [{r['speedup_ci_lo']:.2f},{r['speedup_ci_hi']:.2f}]  "
            f"{r['mean_redundancy']:6.3f}  {r['mean_conflicts']:8.1f}  {r['mean_payload_kb_delivered']:8.1f}"
        )
    free = [e["free_cells"] for e in scaling if e["config"] == "independent" and e["n"] == 1]
    dens = [e["realized_density"] for e in scaling if e["config"] == "independent" and e["n"] == 1]
    print(f"\nrandom maps: {len(free)}, free cells mean {np.mean(free):.0f} (min {min(free)}, max {max(free)}), "
          f"realized density {np.mean(dens):.3f}")
    print("suite overall completion by config:")
    for c in ORDER:
        eps = [e for e in suite if e["config"] == c]
        print(f"  {c:22s} {100*np.mean([e['completed'] for e in eps]):5.1f}%  ({len(eps)} runs)")
    print("total episodes:", len(scaling) + len(suite))


if __name__ == "__main__":
    scaling = load("scaling")
    suite = load("suite")
    rows = summarise(scaling)
    write_summary(rows)
    write_paired_tests(scaling)
    for theme, th in THEMES.items():
        plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": th["ink"]})
        fig_coverage(scaling, th, theme)
        fig_scaling(rows, th, theme)
        fig_suite(suite, th, theme)
    print_headlines(rows, scaling, suite)
