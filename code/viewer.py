import json
import numpy as np
from pathlib import Path
from IPython.display import Image, display

import simulation_parameters as sp
from plots import (
    plot_comparison_goal_reached,
    plot_comparison_nvi,
    plot_comparison_time_to_goal,
    plot_comparison_vulns_backlog,
    plot_comparison_rewards_attacker,
)

_RESULTS_DIR = Path(__file__).parent.parent / 'results'

_SECTION_ORDER = [
    ('goal',     'Goal Reached (%)'),
    ('nvi',      'Mean Network Vulnerability Index (NVI)'),
    ('time',     'Time Metrics'),
    ('rewards',  'Rewards'),
    ('nodes',    'Nodes'),
    ('vulns',    'Vulnerabilities'),
    ('backlog',  'Vulnerability Backlog'),
    ('heatmaps', 'Heatmaps'),
]

_COMPARISON_SECTIONS = [
    ('goal',    'Goal Reached'),
    ('nvi',     'Network Vulnerability Index (NVI)'),
    ('time',    'Time Metrics'),
    ('backlog', 'Vulnerability Backlog'),
    ('rewards', 'Rewards'),
]


# ── Shared utilities ──────────────────────────────────────────────────────────

def find_exp_dir(exp_id: int) -> Path:
    matches = sorted(_RESULTS_DIR.glob(f'exp_{exp_id:02d}__*'))
    if not matches:
        raise FileNotFoundError(
            f'No results folder found for experiment_id={exp_id}.\n'
            f'Run main.py first to generate results.\n'
            f'(Searched in: {_RESULTS_DIR})'
        )
    return matches[0]


def parse_folder_name(folder_name: str) -> dict:
    parts = folder_name.split('__')
    effort_raw = parts[4] if len(parts) > 4 else 'effort_nan'
    return {
        'exp_id'   : int(parts[0].replace('exp_', '')),
        'apt_name' : parts[1] if len(parts) > 1 else '?',
        'campaign' : parts[2] if len(parts) > 2 else '?',
        'network'  : parts[3] if len(parts) > 3 else '?',
        'effort'   : float(effort_raw.replace('effort_', '')),
    }


def _display_pngs(base_dir: Path, sections: list) -> None:
    shown_any = False
    for subfolder, section_title in sections:
        sfpath = base_dir / subfolder
        if not sfpath.exists():
            continue
        pngs = sorted(sfpath.glob('*.png'))
        if not pngs:
            continue
        shown_any = True
        print('\n' + '─' * 60)
        print(section_title)
        print('─' * 60)
        for png in pngs:
            print(f'  {png.stem}')
            display(Image(filename=str(png), width=900))
    return shown_any


# ── Section 1 ─────────────────────────────────────────────────────────────────

def display_experiment(exp_id: int) -> None:
    exp_dir = find_exp_dir(exp_id)
    meta    = parse_folder_name(exp_dir.name)

    print(f"Experiment ID    : {meta['exp_id']:02d}")
    print(f"APT              : {meta['apt_name']}")
    print(f"Campaign         : {meta['campaign']}")
    print(f"Network topology : {meta['network']}")
    print(f"Defender effort  : {meta['effort']}")
    print(f"Results folder   : {exp_dir}")

    root_pngs = sorted(exp_dir.glob('*.png'))
    if root_pngs:
        print('\n' + '─' * 60)
        print('Overview plots')
        print('─' * 60)
        for png in root_pngs:
            display(Image(filename=str(png), width=900))

    shown = _display_pngs(exp_dir, _SECTION_ORDER)
    if not root_pngs and not shown:
        print(f'No PNG plots found in {exp_dir}.\nRun main.py to generate results first.')


# ── Section 2 ─────────────────────────────────────────────────────────────────

class ScenarioData:
    def __init__(self, d):
        self.name                                      = d.get('name', '')
        self.attacker_goal_achieved_per_episode        = np.array(d.get('attacker_goal_achieved_per_episode', []))
        self.rewards_attacker_per_episode              = d.get('rewards_attacker_per_episode', [])
        self.rewards_defender_per_episode              = d.get('rewards_defender_per_episode', [])
        self.num_nodes_exploited_per_episode           = d.get('num_nodes_exploited_per_episode', [])
        self.num_nodes_patched_per_episode             = d.get('num_nodes_patched_per_episode', [])
        self.num_vulnerabilities_exploited_per_episode = d.get('num_vulnerabilities_exploited_per_episode', [])
        self.num_vulnerabilities_patched_per_episode   = d.get('num_vulnerabilities_patched_per_episode', [])
        self.time_to_goal_ATTACKER_per_episode         = d.get('time_to_goal_ATTACKER_per_episode', [])
        self.cumulative_reward_ATTACKER                = d.get('cumulative_reward_ATTACKER', [])
        self.cumulative_reward_DEFENDER                = d.get('cumulative_reward_DEFENDER', [])
        self.nvi_per_episode                           = d.get('nvi_per_episode', [])
        self.mean_vulns_in_backlog_per_episode         = d.get('mean_vulns_in_backlog_per_episode', [])
        self.mean_time_to_patch_per_episode            = d.get('mean_time_to_patch_per_episode', [])
        self.mean_vulns_per_node                       = np.array(d.get('mean_vulns_per_node', []))


def _load_experiment(exp_id: int) -> dict:
    sp.load_experiment(exp_id)
    exp_dir      = find_exp_dir(exp_id)
    scenario_dir = exp_dir / 'scenarios'

    if not scenario_dir.exists():
        raise FileNotFoundError(f'Scenarios folder not found: {scenario_dir}')
    files = sorted(scenario_dir.glob('*.json'))
    if not files:
        raise FileNotFoundError(f'No JSON files found in {scenario_dir}')

    scenarios = {}
    for i, f in enumerate(files, start=1):
        with open(f) as fh:
            scenarios[i] = ScenarioData(json.load(fh))

    return {
        'scenarios'       : scenarios,
        'defender_effort' : 1.0 / sp.DEFENDER_EFFORT,
        'exp_dir'         : str(exp_dir),
        'apt_name'        : sp.APT_NAME,
        'topology'        : sp.TOPOLOGY,
    }


def display_comparison(experiment_ids: list) -> None:
    all_exp = {}
    for eid in experiment_ids:
        all_exp[eid] = _load_experiment(eid)

    print('Loaded experiments:')
    for eid, info in all_exp.items():
        print(f"  EXP {eid:02d} | {info['apt_name']:20s} | {info['topology']:10s} | effort={info['defender_effort']:.4g}")

    comparison_dir = str(_RESULTS_DIR / 'comparisons' / ('exp_' + '__'.join(str(i) for i in experiment_ids)))

    plot_comparison_goal_reached(all_exp, experiment_ids, comparison_dir)
    plot_comparison_nvi(all_exp, experiment_ids, comparison_dir)
    plot_comparison_time_to_goal(all_exp, experiment_ids, comparison_dir)
    plot_comparison_vulns_backlog(all_exp, experiment_ids, comparison_dir)
    plot_comparison_rewards_attacker(all_exp, experiment_ids, comparison_dir)

    print(f'\nPlots saved to: {comparison_dir}')
    _display_pngs(Path(comparison_dir), _COMPARISON_SECTIONS)
