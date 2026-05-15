import os
import numpy as np
from collections import defaultdict
from Scenario import SCENARIOS

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESULTS_MD = os.path.join(_ROOT, "results", "RESULTS.md")

_DEFENDER_ORDER = [None, "importance", "severity", "centrality"]
_SCENARIO_BY_DEFENDER = {cfg["defender"]: sid for sid, cfg in SCENARIOS.items()}


def _metrics(scenario):
    s = scenario
    att_r = round(np.mean(s.rewards_attacker_per_episode), 2) if s.rewards_attacker_per_episode else "-"
    def_r = round(np.mean(s.rewards_defender_per_episode), 2) if s.rewards_defender_per_episode else "-"
    goal  = round(np.mean(s.attacker_goal_achieved_per_episode) * 100)
    nvi   = round(np.mean(s.nvi_per_episode)) if s.nvi_per_episode else "-"
    ttrg  = round(np.mean(s.time_to_goal_ATTACKER_per_episode)) if s.time_to_goal_ATTACKER_per_episode else "-"
    vib   = round(np.mean(s.mean_vulns_in_backlog_per_episode)) if s.mean_vulns_in_backlog_per_episode else "-"
    ttpv  = round(np.mean(s.mean_time_to_patch_per_episode)) if s.mean_time_to_patch_per_episode else "-"
    return att_r, def_r, goal, nvi, ttrg, vib, ttpv


def _row(defender_label, effort, att_r, def_r, goal, nvi, ttrg, vib, ttpv, first_in_group=False):
    col1 = f"**{defender_label}**" if first_in_group else ""
    return f"| {col1:<18} | {effort:<7} | {att_r:<17} | {def_r:<17} | {goal:<20} | {nvi:<7} | {ttrg:<16} | {vib:<8} | {ttpv:<16} |\n"


def _table(group_experiments):
    header = (
        "| Defender           | Effort  | Attacker's Reward | Defender's Reward "
        "| Goal Achievement (%) | NVI (%) | Mean TTRG (days) | Mean VIB | Mean TTPV (days) |\n"
        "|--------------------|---------|--------------------|--------------------"
        "|----------------------|---------|------------------|----------|------------------|\n"
    )
    rows = [header]

    # Sort experiments by DEFENDER_EFFORT ascending (Low → High)
    sorted_exps = sorted(group_experiments, key=lambda e: e["defender_effort"])

    for defender in _DEFENDER_ORDER:
        sid = _SCENARIO_BY_DEFENDER.get(defender)
        if sid is None:
            continue
        defender_label = "None" if defender is None else defender.capitalize()

        if defender is None:
            # No defender: effort is irrelevant, use first experiment only
            exp = sorted_exps[0]
            scenario = exp["scenarios"].get(sid)
            if scenario is None:
                continue
            att_r, def_r, goal, nvi, ttrg, vib, ttpv = _metrics(scenario)
            rows.append(_row(defender_label, "-", att_r, def_r, goal, nvi, ttrg, vib, ttpv, first_in_group=True))
        else:
            for i, exp in enumerate(sorted_exps):
                scenario = exp["scenarios"].get(sid)
                if scenario is None:
                    continue
                att_r, def_r, goal, nvi, ttrg, vib, ttpv = _metrics(scenario)
                effort = exp["defender_effort"]
                rows.append(_row(defender_label, effort, att_r, def_r, goal, nvi, ttrg, vib, ttpv, first_in_group=(i == 0)))

    return "".join(rows)


def update_results_md(all_experiments):
    # Group by (apt_name, topology)
    groups = defaultdict(list)
    for exp in all_experiments.values():
        key = (exp["apt_name"], exp["topology"])
        groups[key].append(exp)

    lines = ["# Attacker\n"]

    current_apt = None
    for (apt_name, topology), group_exps in sorted(groups.items()):
        if apt_name != current_apt:
            lines.append(f"\n## {apt_name}\n")
            current_apt = apt_name
        lines.append(f"### {topology.capitalize()} Topology\n\n")
        lines.append(_table(group_exps))
        lines.append("\n")

    with open(_RESULTS_MD, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"RESULTS.md updated at {_RESULTS_MD}.")
