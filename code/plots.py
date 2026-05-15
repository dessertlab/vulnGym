import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import networkx as nx

from Scenario import SCENARIOS

# ── Palette ───────────────────────────────────────────────────────────────────

POLICY_COLORS = {
    "None":       "#808080",
    "Importance": "#2D7FF9",
    "Severity":   "#FF9F1C",
    "Centrality": "#8E6CFF",
    "Random":     "#E74C3C",
}

ZONE_COLORS = {
    "external": "#DFEAF6",
    "dmz":      "#CBF0CB",
    "internal": "#F6E9AA",
    "database": "#F19C9B",
}

# ── Internal helpers ──────────────────────────────────────────────────────────

def _label(scenario_id: int) -> str:
    d = SCENARIOS[scenario_id]["defender"]
    return "None" if d is None else d.capitalize()

def _color(label: str) -> str:
    return POLICY_COLORS.get(label, "#555555")

def _save(fig: plt.Figure, directory: str, filename: str) -> None:
    os.makedirs(directory, exist_ok=True)
    fig.savefig(os.path.join(directory, f"{filename}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

def _mean(lst) -> float:
    return float(np.nan) if not lst else float(np.mean(lst))

def _std(lst) -> float:
    return float(np.nan) if not lst else float(np.std(lst))

def _bar(labels, values, stds=None, ylabel="", ylim=None,
         figsize=(10, 6)) -> plt.Figure:
    """Thin wrapper for a single-series bar chart with optional error bars."""
    x = np.arange(len(labels))
    colors = [_color(l) for l in labels]
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x, values, width=0.5, color=colors, yerr=stds, capsize=5, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylabel(ylabel, fontsize=14)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", linestyle="--", alpha=0.7, zorder=0)
    fig.tight_layout()
    return fig

def _legend_patches(labels) -> list:
    return [mpatches.Patch(color=_color(l), label=l) for l in labels]


# ── 1. Goal reached (%) ───────────────────────────────────────────────────────

def plot_goal_reached(scenarios: dict, exp_dir: str) -> None:
    """Bar chart: percentage of episodes in which the attacker reached the goal."""
    labels, values = [], []
    for sid in sorted(SCENARIOS):
        goals = scenarios[sid].attacker_goal_achieved_per_episode
        labels.append(_label(sid))
        values.append(100 * sum(goals) / len(goals))

    fig = _bar(labels, values, ylabel="Goal reached (%)", ylim=(0, 100))
    fig.axes[0].set_xlabel("Defender policy", fontsize=14)
    _save(fig, f"{exp_dir}/goal", "goal_reached")


# ── 2. Time to goal ───────────────────────────────────────────────────────────

def plot_time_to_goal(scenarios: dict, exp_dir: str) -> None:
    """Bar chart: mean ± std simulation steps for the attacker to reach the goal."""
    labels, values, stds = [], [], []
    for sid in sorted(SCENARIOS):
        data = scenarios[sid].time_to_goal_ATTACKER_per_episode
        labels.append(_label(sid))
        values.append(_mean(data))
        stds.append(_std(data))

    fig = _bar(labels, values, stds, ylabel="Avg steps to goal")
    fig.axes[0].set_xlabel("Defender policy", fontsize=14)
    _save(fig, f"{exp_dir}/time", "time_to_goal")


# ── 3. NVI (Network Vulnerability Index) ─────────────────────────────────────

def plot_nvi(scenarios: dict, exp_dir: str) -> None:
    """Bar chart: mean ± std NVI (%) per scenario."""
    labels, values, stds = [], [], []
    for sid in sorted(SCENARIOS):
        data = scenarios[sid].nvi_per_episode
        labels.append(_label(sid))
        values.append(_mean(data))
        stds.append(_std(data))

    fig = _bar(labels, values, stds,
               ylabel="Mean Network Vulnerability Index (%)", ylim=(0, 100))
    fig.axes[0].set_xlabel("Defender policy", fontsize=14)
    _save(fig, f"{exp_dir}/nvi", "nvi")



# ── 5. Attacker reward per episode ───────────────────────────────────────────

def plot_rewards_attacker(scenarios: dict, exp_dir: str) -> None:
    """Line chart: attacker reward per episode for each scenario."""
    fig, ax = plt.subplots(figsize=(12, 5))
    for sid in sorted(SCENARIOS):
        lbl = _label(sid)
        ax.plot(scenarios[sid].rewards_attacker_per_episode,
                label=f"vs {lbl} Defender", color=_color(lbl))
    ax.set_xlabel("Episode", fontsize=14)
    ax.set_ylabel("Reward", fontsize=14)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    _save(fig, f"{exp_dir}/rewards", "rewards_attacker_per_episode")


# ── 6. Defender reward per episode ───────────────────────────────────────────

def plot_rewards_defender(scenarios: dict, exp_dir: str) -> None:
    """Line chart: defender reward per episode for each defender scenario."""
    fig, ax = plt.subplots(figsize=(12, 5))
    for sid in sorted(SCENARIOS):
        if SCENARIOS[sid]["defender"] is None:
            continue
        lbl = _label(sid)
        ax.plot(scenarios[sid].rewards_defender_per_episode,
                label=lbl, color=_color(lbl))
    ax.set_xlabel("Episode", fontsize=14)
    ax.set_ylabel("Reward", fontsize=14)
    ax.legend(title="Defender policy", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    _save(fig, f"{exp_dir}/rewards", "rewards_defender_per_episode")





# ── 9. Mean reward per scenario (bar) ────────────────────────────────────────

def plot_mean_rewards(scenarios: dict, exp_dir: str) -> None:
    """Grouped bar chart: mean attacker and defender reward per scenario."""
    labels, atk_vals, atk_stds, def_vals, def_stds = [], [], [], [], []
    for sid in sorted(SCENARIOS):
        labels.append(_label(sid))
        atk_vals.append(_mean(scenarios[sid].rewards_attacker_per_episode))
        atk_stds.append(_std(scenarios[sid].rewards_attacker_per_episode))
        def_vals.append(_mean(scenarios[sid].rewards_defender_per_episode))
        def_stds.append(_std(scenarios[sid].rewards_defender_per_episode))

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - w / 2, atk_vals, w, yerr=atk_stds, capsize=4,
           label="Attacker", color="#E74C3C", zorder=3)
    ax.bar(x + w / 2, def_vals, w, yerr=def_stds, capsize=4,
           label="Defender", color="#2D7FF9", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14)
    ax.set_ylabel("Mean reward per episode", fontsize=14)
    ax.set_xlabel("Defender policy", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.7, zorder=0)
    fig.tight_layout()
    _save(fig, f"{exp_dir}/rewards", "mean_rewards")


# ── 10. Nodes exploited vs patched (bar) ──────────────────────────────────────

def plot_nodes_exploited_patched(scenarios: dict, exp_dir: str) -> None:
    """Grouped bar chart: mean exploited and patched nodes per scenario."""
    labels, exp_vals, exp_stds, pat_vals, pat_stds = [], [], [], [], []
    for sid in sorted(SCENARIOS):
        labels.append(_label(sid))
        exp_vals.append(_mean(scenarios[sid].num_nodes_exploited_per_episode))
        exp_stds.append(_std(scenarios[sid].num_nodes_exploited_per_episode))
        pat_vals.append(_mean(scenarios[sid].num_nodes_patched_per_episode))
        pat_stds.append(_std(scenarios[sid].num_nodes_patched_per_episode))

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - w / 2, exp_vals, w, yerr=exp_stds, capsize=4,
           label="Exploited", color="#E74C3C", zorder=3)
    ax.bar(x + w / 2, pat_vals, w, yerr=pat_stds, capsize=4,
           label="Patched", color="#2D7FF9", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14)
    ax.set_ylabel("Mean nodes per episode", fontsize=14)
    ax.set_xlabel("Defender policy", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.7, zorder=0)
    fig.tight_layout()
    _save(fig, f"{exp_dir}/nodes", "nodes_exploited_patched")



# ── 12. Vulnerabilities exploited vs patched (bar) ───────────────────────────

def plot_vulns_exploited_patched(scenarios: dict, exp_dir: str) -> None:
    """Grouped bar chart: mean exploited and patched vulnerabilities per scenario."""
    labels, exp_vals, exp_stds, pat_vals, pat_stds = [], [], [], [], []
    for sid in sorted(SCENARIOS):
        labels.append(_label(sid))
        exp_vals.append(_mean(scenarios[sid].num_vulnerabilities_exploited_per_episode))
        exp_stds.append(_std(scenarios[sid].num_vulnerabilities_exploited_per_episode))
        pat_vals.append(_mean(scenarios[sid].num_vulnerabilities_patched_per_episode))
        pat_stds.append(_std(scenarios[sid].num_vulnerabilities_patched_per_episode))

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - w / 2, exp_vals, w, yerr=exp_stds, capsize=4,
           label="Exploited", color="#E74C3C", zorder=3)
    ax.bar(x + w / 2, pat_vals, w, yerr=pat_stds, capsize=4,
           label="Patched", color="#2D7FF9", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14)
    ax.set_ylabel("Mean vulnerabilities per episode", fontsize=14)
    ax.set_xlabel("Defender policy", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.7, zorder=0)
    fig.tight_layout()
    _save(fig, f"{exp_dir}/vulns", "vulns_exploited_patched")


# ── 14. Vulnerability backlog ─────────────────────────────────────────────────

def plot_vulns_backlog(scenarios: dict, exp_dir: str) -> None:
    """Bar chart: mean ± std vulnerabilities in the defender's backlog."""
    labels, values, stds = [], [], []
    for sid in sorted(SCENARIOS):
        if SCENARIOS[sid]["defender"] is None:
            continue
        data = scenarios[sid].mean_vulns_in_backlog_per_episode
        labels.append(_label(sid))
        values.append(_mean(data))
        stds.append(_std(data))

    fig = _bar(labels, values, stds, ylabel="Avg vulnerabilities in backlog")
    fig.axes[0].set_xlabel("Defender policy", fontsize=14)
    _save(fig, f"{exp_dir}/backlog", "vulns_backlog")


# ── 16. Time to patch ─────────────────────────────────────────────────────────

def plot_time_to_patch(scenarios: dict, exp_dir: str) -> None:
    """Bar chart: mean ± std patching time (simulation days) per defender policy."""
    labels, values, stds = [], [], []
    for sid in sorted(SCENARIOS):
        if SCENARIOS[sid]["defender"] is None:
            continue
        data = scenarios[sid].mean_time_to_patch_per_episode
        labels.append(_label(sid))
        values.append(_mean(data))
        stds.append(_std(data))

    fig = _bar(labels, values, stds, ylabel="Avg days to patch a vulnerability")
    fig.axes[0].set_xlabel("Defender policy", fontsize=14)
    _save(fig, f"{exp_dir}/time", "time_to_patch")


# ── 19. Attacker & defender node-visit heatmaps ───────────────────────────────

def _node_positions(env) -> dict:
    """Compute consistent x/y positions for nodes, grouped by network zone."""
    G = env.network
    zone_order = ["external", "dmz", "internal", "database"]
    y_levels   = {"external": 1, "dmz": 4, "internal": 7, "database": 10}
    zones      = {n: G.nodes[n]["node"].zone for n in G.nodes}

    positions = {}
    base_spacing = 12.0
    min_width    = 200.0

    for zone in zone_order:
        nodes_in_zone = sorted(n for n, z in zones.items() if z == zone)
        k = len(nodes_in_zone)
        if k == 0:
            continue
        if k == 1:
            positions[nodes_in_zone[0]] = (0, y_levels[zone])
        else:
            width = max(min_width, base_spacing * (k - 1))
            for i, node in enumerate(nodes_in_zone):
                positions[node] = (
                    np.linspace(-width / 2, width / 2, k)[i],
                    y_levels[zone]
                )
    return positions


def _draw_zone_bands(ax, y_levels: dict, zone_boundaries: list,
                     zone_order: list) -> None:
    """Fill each zone with a translucent background band."""
    for i, zone in enumerate(zone_order):
        y = y_levels[zone]
        y_bot = zone_boundaries[i - 1] if i > 0 else y - (zone_boundaries[0] - y)
        y_top = zone_boundaries[i]     if i < len(zone_order) - 1 \
                else y + (y - zone_boundaries[-1])
        ax.axhspan(y_bot, y_top, color=ZONE_COLORS[zone], alpha=0.35, zorder=0)
    for yb in zone_boundaries:
        ax.axhline(yb, color="black", linestyle="--", linewidth=0.8,
                   alpha=0.4, zorder=1)


def _heatmap_single(G, node_positions, node_list, values, cmap,
                    cbar_label: str, zone_order: list,
                    y_levels: dict, zone_boundaries: list,
                    x_min: float, x_max: float) -> plt.Figure:
    """Render one heatmap figure (attacker or defender)."""
    fig, ax = plt.subplots(figsize=(20, 5))
    _draw_zone_bands(ax, y_levels, zone_boundaries, zone_order)

    nodes_sc = nx.draw_networkx_nodes(
        G, node_positions, nodelist=node_list,
        node_color=values, cmap=cmap, node_size=500,
        edgecolors="black", linewidths=0.6, ax=ax,
    )
    nx.draw_networkx_edges(G, node_positions, alpha=0.3, width=1.0, ax=ax)
    nx.draw_networkx_labels(G, node_positions, font_size=10, ax=ax)

    ax.set_xlim(x_min - 8, x_max + 8)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(nodes_sc, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(cbar_label, fontsize=12)

    legend_patches = [
        mpatches.Patch(
            facecolor=ZONE_COLORS[z], alpha=0.35,
            label="DMZ" if z == "dmz" else z.capitalize()
        )
        for z in zone_order
    ]
    ax.legend(handles=legend_patches, title="Zone",
              loc="upper center", bbox_to_anchor=(0.5, 1.15),
              ncol=len(zone_order), frameon=True, facecolor="white",
              framealpha=1, fancybox=False, fontsize=12, title_fontsize=13)

    fig.tight_layout()
    return fig


def plot_heatmaps(scenarios: dict, exp_dir: str) -> None:
    """
    For each defender scenario, save two heatmaps:
      - attacker mean node visits  (Reds colormap)
      - defender mean node visits  (Blues colormap)
    """
    zone_order  = ["external", "dmz", "internal", "database"]
    y_levels    = {"external": 1, "dmz": 4, "internal": 7, "database": 10}
    zone_boundaries = [
        (y_levels[zone_order[i]] + y_levels[zone_order[i + 1]]) / 2
        for i in range(len(zone_order) - 1)
    ]

    for sid in sorted(SCENARIOS):
        if SCENARIOS[sid]["defender"] is None:
            continue

        scenario = scenarios[sid]
        lbl      = _label(sid).lower()
        env      = scenario.environment
        G        = env.network
        node_pos = _node_positions(env)
        node_list = list(G.nodes())

        all_x = [p[0] for p in node_pos.values()]
        x_min, x_max = min(all_x), max(all_x)

        atk_vals = [scenario.attacker_mean_node_visits[n] for n in node_list]
        def_vals = [scenario.defender_mean_node_visits[n] for n in node_list]

        out_dir = f"{exp_dir}/heatmaps"

        fig_a = _heatmap_single(
            G, node_pos, node_list, atk_vals, plt.cm.Reds,
            "Mean attacker visits", zone_order, y_levels,
            zone_boundaries, x_min, x_max
        )
        _save(fig_a, out_dir, f"attacker_{lbl}_heatmap")

        fig_d = _heatmap_single(
            G, node_pos, node_list, def_vals, plt.cm.Blues,
            "Mean defender visits", zone_order, y_levels,
            zone_boundaries, x_min, x_max
        )
        _save(fig_d, out_dir, f"defender_{lbl}_heatmap")


# ── 20. Attacker-only heatmap (no-defender scenario) ─────────────────────────

def plot_heatmap_no_defender(scenarios: dict, exp_dir: str) -> None:
    """
    Heatmap of attacker mean visits for the no-defender baseline scenario.
    """
    baseline_sid = next(
        (sid for sid in sorted(SCENARIOS) if SCENARIOS[sid]["defender"] is None),
        None
    )
    if baseline_sid is None:
        return

    scenario = scenarios[baseline_sid]
    env      = scenario.environment
    G        = env.network
    node_pos = _node_positions(env)
    node_list = list(G.nodes())

    zone_order  = ["external", "dmz", "internal", "database"]
    y_levels    = {"external": 1, "dmz": 4, "internal": 7, "database": 10}
    zone_boundaries = [
        (y_levels[zone_order[i]] + y_levels[zone_order[i + 1]]) / 2
        for i in range(len(zone_order) - 1)
    ]

    all_x = [p[0] for p in node_pos.values()]
    x_min, x_max = min(all_x), max(all_x)

    atk_vals = [scenario.attacker_mean_node_visits[n] for n in node_list]

    fig = _heatmap_single(
        G, node_pos, node_list, atk_vals, plt.cm.Reds,
        "Mean attacker visits", zone_order, y_levels,
        zone_boundaries, x_min, x_max
    )
    _save(fig, f"{exp_dir}/heatmaps", "attacker_no_defender_heatmap")


# ── Master function ───────────────────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════════════════════════
# COMPARISON PLOTS  (across multiple experiments / defender-effort levels)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Signature for all comparison functions:
#   all_experiments : dict  {exp_id: {"scenarios": dict,
#                                     "defender_effort": float,  (= 1/DEFENDER_EFFORT)
#                                     "exp_dir": str}}
#   experiment_ids  : list  ordered list of exp IDs to compare
#   save_dir        : str   e.g. "../results/comparisons"
#
# Layout:
#   x = 0      → standalone baseline bar (Scenario 1, no defender)
#   x = 1 … N  → one group per experiment, bars per defender policy
# ───────────────────────────────────────────────────────────────────────────────

def _effort_label(e: float) -> str:
    """Map defender_effort (= 1 / DEFENDER_EFFORT) to a readable x-axis label."""
    if e > 1:
        return f"High"
    elif e == 1.0:
        return "Regular"
    else:
        return f"Low"


def _comparison_fig(
    all_experiments: dict,
    experiment_ids: list,
    extract: callable,
    extract_err: callable = None,
    ylabel: str = "",
    ylim: tuple = None,
    figsize: tuple = (13, 6),
) -> plt.Figure:
    """
    Generic grouped-bar comparison across effort levels.
    Slot 0: Scenario 1 baseline (no defender) from experiment_ids[0].
    Slots 1+: one slot per experiment_id, bars per defender policy.
    """
    defender_sids = [sid for sid in sorted(SCENARIOS)
                     if SCENARIOS[sid]["defender"] is not None]
    efforts   = [all_experiments[eid]["defender_effort"] for eid in experiment_ids]
    n_bars    = len(defender_sids)
    bar_width = 0.7 / n_bars
    x_pos     = np.arange(len(experiment_ids) + 1)

    fig, ax = plt.subplots(figsize=figsize)

    # baseline bar (Scenario 1) — only if the metric is defined for no-defender
    sc1      = all_experiments[experiment_ids[0]]["scenarios"][1]
    val_base = extract(sc1)
    show_baseline = not (isinstance(val_base, float) and np.isnan(val_base))
    if show_baseline:
        err_base = extract_err(sc1) if extract_err else None
        ax.bar(0, val_base, width=0.4, color=_color("None"),
               yerr=err_base, capsize=4, label="None", zorder=3)

    # one bar-group per experiment
    for i, sid in enumerate(defender_sids):
        lbl    = _label(sid)
        values = [extract(all_experiments[eid]["scenarios"][sid]) for eid in experiment_ids]
        errors = ([extract_err(all_experiments[eid]["scenarios"][sid]) for eid in experiment_ids]
                  if extract_err else None)
        offset = (i - n_bars / 2) * bar_width + bar_width / 2
        ax.bar(x_pos[1:] + offset, values, bar_width,
               color=_color(lbl), yerr=errors, capsize=4, label=lbl, zorder=3)

    x_ticks  = x_pos if show_baseline else x_pos[1:]
    x_labels = (["None"] if show_baseline else []) + [_effort_label(e) for e in efforts]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=13)
    ax.set_xlabel("Defender effort", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", linestyle="--", alpha=0.7, zorder=0)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),
              title="Defender policy",
              loc="upper center", bbox_to_anchor=(0.5, 1.15),
              ncol=len(by_label), fontsize=12, title_fontsize=13)

    fig.tight_layout()
    return fig


def plot_comparison_goal_reached(
        all_experiments: dict, experiment_ids: list, save_dir: str) -> None:
    """Grouped bar: % episodes goal reached, across defender-effort levels."""
    def _val(sc):
        g = sc.attacker_goal_achieved_per_episode
        return 100 * sum(g) / len(g)

    fig = _comparison_fig(all_experiments, experiment_ids, _val,
                          ylabel="Goal reached (%)", ylim=(0, 100))
    _save(fig, f"{save_dir}/goal", "goal_reached_comparison")


def plot_comparison_nvi(
        all_experiments: dict, experiment_ids: list, save_dir: str) -> None:
    """Grouped bar: mean NVI (%) across defender-effort levels."""
    fig = _comparison_fig(
        all_experiments, experiment_ids,
        extract     = lambda sc: _mean(sc.nvi_per_episode),
        extract_err = lambda sc: _std(sc.nvi_per_episode),
        ylabel="Mean NVI (%)", ylim=(0, 100),
    )
    _save(fig, f"{save_dir}/nvi", "nvi_comparison")


def plot_comparison_time_to_goal(
        all_experiments: dict, experiment_ids: list, save_dir: str) -> None:
    """Grouped bar: mean steps to goal across defender-effort levels."""
    fig = _comparison_fig(
        all_experiments, experiment_ids,
        extract     = lambda sc: _mean(sc.time_to_goal_ATTACKER_per_episode),
        extract_err = lambda sc: _std(sc.time_to_goal_ATTACKER_per_episode),
        ylabel="Avg steps to goal",
    )
    _save(fig, f"{save_dir}/time", "time_to_goal_comparison")


def plot_comparison_vulns_backlog(
        all_experiments: dict, experiment_ids: list, save_dir: str) -> None:
    """Grouped bar: mean vulnerability backlog across defender-effort levels."""
    fig = _comparison_fig(
        all_experiments, experiment_ids,
        extract     = lambda sc: _mean(sc.mean_vulns_in_backlog_per_episode),
        extract_err = lambda sc: _std(sc.mean_vulns_in_backlog_per_episode),
        ylabel="Avg vulns in backlog",
    )
    _save(fig, f"{save_dir}/backlog", "vulns_backlog_comparison")




def plot_comparison_rewards_attacker(
        all_experiments: dict, experiment_ids: list, save_dir: str) -> None:
    """Grouped bar: mean attacker reward per episode across defender-effort levels."""
    fig = _comparison_fig(
        all_experiments, experiment_ids,
        extract     = lambda sc: _mean(sc.rewards_attacker_per_episode),
        extract_err = lambda sc: _std(sc.rewards_attacker_per_episode),
        ylabel="Mean attacker reward per episode",
    )
    _save(fig, f"{save_dir}/rewards", "rewards_attacker_comparison")


def plot_all_comparisons(
        all_experiments: dict, experiment_ids: list, save_dir: str) -> None:
    """
    Save all comparison plots for a group of experiments.

    Args:
        all_experiments : dict  {exp_id: {"scenarios": dict,
                                          "defender_effort": float,
                                          "exp_dir": str}}
        experiment_ids  : list  e.g. [12, 11, 10, 9]  (same APT / topology)
        save_dir        : str   e.g. "../results/comparisons"
    """
    plot_comparison_goal_reached(all_experiments, experiment_ids, save_dir)
    plot_comparison_nvi(all_experiments, experiment_ids, save_dir)
    plot_comparison_time_to_goal(all_experiments, experiment_ids, save_dir)
    plot_comparison_vulns_backlog(all_experiments, experiment_ids, save_dir)
    plot_comparison_rewards_attacker(all_experiments, experiment_ids, save_dir)


# ── 21. Network topology ─────────────────────────────────────────────────────

def plot_network(env, exp_dir: str) -> None:
    if env.topology == "layered":
        plot_network_layered(env, exp_dir)
    elif env.topology == "tree":
        plot_network_tree(env, exp_dir)
    else:
        raise ValueError(f"Unknown topology: {env.topology}")


def plot_network_layered(env, exp_dir: str) -> None:
    zones = {node: env.network.nodes[node]["node"].zone for node in env.network.nodes}
    node_positions = {}

    zone_order = ["external", "dmz", "internal", "database"]
    y_levels   = {"external": 1, "dmz": 4, "internal": 7, "database": 10}
    colors     = {
        "external": "#8DB3E2",
        "dmz":      "#88C999",
        "internal": "#E8B37A",
        "database": "#D98C8C",
    }

    width_limit = 40.0
    for zone in zone_order:
        nodes_in_zone = sorted(n for n, z in zones.items() if z == zone)
        k = len(nodes_in_zone)
        if k == 0:
            continue
        if k == 1:
            node_positions[nodes_in_zone[0]] = (0, y_levels[zone])
        else:
            for i, node in enumerate(nodes_in_zone):
                node_positions[node] = (
                    np.linspace(-width_limit, width_limit, k)[i],
                    y_levels[zone],
                )

    fig, ax = plt.subplots(figsize=(20, 8))
    node_size = 500

    for zone, color in colors.items():
        nodes_in_zone = [n for n in env.network.nodes() if zones.get(n) == zone]
        shared = [n for n in nodes_in_zone if getattr(env.network.nodes[n]["node"], "is_shared", False)]
        normal = [n for n in nodes_in_zone if n not in shared]

        if normal:
            nx.draw_networkx_nodes(env.network, node_positions, nodelist=normal,
                                   node_color=color, node_size=node_size, alpha=0.95, ax=ax)
        if shared:
            nx.draw_networkx_nodes(env.network, node_positions, nodelist=shared,
                                   node_color=color, node_size=node_size, alpha=0.95,
                                   edgecolors="red", linewidths=1.5, ax=ax)

    nx.draw_networkx_edges(env.network, node_positions, alpha=0.3, width=1.0, ax=ax)
    nx.draw_networkx_labels(env.network, node_positions, font_size=10, ax=ax)

    for zone in zone_order:
        ax.axhline(y=y_levels[zone], color="gray", linestyle="--", alpha=0.2)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", label=zone.capitalize(),
               markerfacecolor=color, markersize=10)
        for zone, color in colors.items()
    ]
    ax.legend(handles=legend_handles, title="Zone",
              loc="upper center", bbox_to_anchor=(0.5, 1.05),
              ncol=4, frameon=True, fontsize=14, title_fontsize=16)

    ax.set_aspect("auto", adjustable="datalim")
    ax.margins(x=0.02, y=0.1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    _save(fig, exp_dir, "network_layered")


def plot_network_tree(env, exp_dir: str) -> None:
    zones = {node: env.network.nodes[node]["node"].zone for node in env.network.nodes}
    node_positions = {}

    zone_order = ["external", "dmz", "internal", "database"]
    y_levels   = {"external": 1, "dmz": 4, "internal": 7, "database": 10}
    colors     = {
        "external": "#8DB3E2",
        "dmz":      "#88C999",
        "internal": "#E8B37A",
        "database": "#D98C8C",
    }
    zone_rank = {"external": 0, "dmz": 1, "internal": 2, "database": 3}

    width_limit = 40.0
    for zone in zone_order:
        nodes_in_zone = sorted(n for n, z in zones.items() if z == zone)
        k = len(nodes_in_zone)
        if k == 0:
            continue
        if k == 1:
            node_positions[nodes_in_zone[0]] = (0, y_levels[zone])
        else:
            for i, node in enumerate(nodes_in_zone):
                node_positions[node] = (
                    np.linspace(-width_limit, width_limit, k)[i],
                    y_levels[zone],
                )

    gateway_nodes = set()
    for u, v, data in env.network.edges(data=True):
        if data.get("layer_edge", False):
            gateway_nodes.add(u if zone_rank[zones[u]] < zone_rank[zones[v]] else v)

    fig, ax = plt.subplots(figsize=(20, 8))
    node_size = 500

    for zone, color in colors.items():
        nodes_in_zone = [n for n in env.network.nodes() if zones.get(n) == zone]
        gateway = [n for n in nodes_in_zone if n in gateway_nodes]
        normal  = [n for n in nodes_in_zone if n not in gateway]

        if normal:
            nx.draw_networkx_nodes(env.network, node_positions, nodelist=normal,
                                   node_color=color, node_size=node_size, alpha=0.95, ax=ax)
        if gateway:
            nx.draw_networkx_nodes(env.network, node_positions, nodelist=gateway,
                                   node_color=color, node_size=node_size, alpha=0.95,
                                   edgecolors="blue", linewidths=2.5, ax=ax)

    intra_edges = [(u, v) for u, v, d in env.network.edges(data=True) if not d.get("layer_edge", False)]
    inter_edges = [(u, v) for u, v, d in env.network.edges(data=True) if d.get("layer_edge", False)]

    nx.draw_networkx_edges(env.network, node_positions, edgelist=intra_edges, alpha=0.2, width=1.0, ax=ax)
    nx.draw_networkx_edges(env.network, node_positions, edgelist=inter_edges, alpha=0.3, width=1.2, ax=ax)
    nx.draw_networkx_labels(env.network, node_positions, font_size=10, ax=ax)

    for zone in zone_order:
        ax.axhline(y=y_levels[zone], color="gray", linestyle="--", alpha=0.2)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", label=zone.capitalize(),
               markerfacecolor=color, markersize=10)
        for zone, color in colors.items()
    ]
    legend_handles.append(
        Line2D([0], [0], marker="o", color="w", label="Gateway Node",
               markerfacecolor="white", markeredgecolor="blue", markersize=10)
    )
    ax.legend(handles=legend_handles, title="Zone",
              loc="upper center", bbox_to_anchor=(0.5, 1.05),
              ncol=5, frameon=True, fontsize=14, title_fontsize=16)

    ax.set_aspect("auto", adjustable="datalim")
    ax.margins(x=0.02, y=0.1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    _save(fig, exp_dir, "network_tree")


# ── Per-experiment master ──────────────────────────────────────────────────────

def plot_all(scenarios: dict, exp_dir: str) -> None:
    """
    Generate and save every plot for one experiment.

    Args:
        scenarios: dict mapping scenario_id → Scenario object (after run_scenario)
        exp_dir:   output directory (e.g. returned by plotting.get_experiment_dir)
    """
    # Network topology
    first_env = scenarios[min(scenarios)].environment
    plot_network(first_env, exp_dir)

    # Summary bar charts
    plot_goal_reached(scenarios, exp_dir)
    plot_time_to_goal(scenarios, exp_dir)
    plot_nvi(scenarios, exp_dir)
    plot_vulns_backlog(scenarios, exp_dir)
    plot_time_to_patch(scenarios, exp_dir)
    plot_mean_rewards(scenarios, exp_dir)
    plot_nodes_exploited_patched(scenarios, exp_dir)
    plot_vulns_exploited_patched(scenarios, exp_dir)

    # Per-episode line charts
    plot_rewards_attacker(scenarios, exp_dir)
    plot_rewards_defender(scenarios, exp_dir)

    # Network heatmaps
    plot_heatmap_no_defender(scenarios, exp_dir)
    plot_heatmaps(scenarios, exp_dir)
