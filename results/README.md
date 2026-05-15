# Results

This folder is populated automatically by `code/main.py` each time experiments are run.

---

## Folder Structure

```
results/
├── README.md                                  
├── RESULTS.md                                  # Auto-generated summary table of all experiments
├── comparisons/                                # Cross-experiment comparison plots
│   └── exp_<N1>__<N2>__<N3>/                  # One subfolder per group of compared experiments
│       ├── rewards/
│       ├── goal/
│       ├── nvi/
│       ├── backlog/
│       └── time/
└── exp_<NN>__<APT>__<Campaign>__<Topology>__effort_<X>/   # One folder per experiment
    ├── network_layered.png | network_tree.png  # Network topology visualization
    ├── scenarios/                              # Raw JSON data for each scenario
    │   ├── Scenario_1:_DQN_Attacker_-_No_Defender.json
    │   ├── Scenario_2:_DQN_Attacker_-_Policy_Defender_(Importance).json
    │   ├── Scenario_3:_DQN_Attacker_-_Policy_Defender_(Severity).json
    │   └── Scenario_4:_DQN_Attacker_-_Policy_Defender_(Centrality).json
    ├── heatmaps/                               # Node-level exploitation and patching heatmaps
    │   ├── attacker_no_defender_heatmap.png
    │   ├── attacker_importance_heatmap.png
    │   ├── attacker_severity_heatmap.png
    │   ├── attacker_centrality_heatmap.png
    │   ├── defender_importance_heatmap.png
    │   ├── defender_severity_heatmap.png
    │   └── defender_centrality_heatmap.png
    ├── rewards/                                # Reward curves over episodes
    │   ├── rewards_attacker_per_episode.png
    │   ├── rewards_defender_per_episode.png
    │   └── mean_rewards.png
    ├── goal/                                   # Goal achievement rate across scenarios
    │   └── goal_reached.png
    ├── nvi/                                    # Network Vulnerability Index over time
    │   └── nvi.png
    ├── nodes/                                  # Nodes exploited vs. patched
    │   └── nodes_exploited_patched.png
    ├── vulns/                                  # Vulnerabilities exploited vs. patched
    │   └── vulns_exploited_patched.png
    ├── backlog/                                # Defender vulnerability backlog over time
    │   └── vulns_backlog.png
    └── time/                                   # Timing metrics
        ├── time_to_goal.png
        └── time_to_patch.png
```

---

## Experiment Folder Naming

Each experiment folder follows the pattern:

```
exp_<NN>__<APT>__<Campaign>__<Topology>__effort_<X>
```

| Field | Example values |
| --- | --- |
| `NN` | Zero-padded experiment ID: `01`, `02`, ... |
| `APT` | APT group name: `APT28`, `APT41` |
| `Campaign` | Attack campaign: `DoS`, `Impact` |
| `Topology` | Network topology: `Layered`, `Tree` |
| `X` | Defender effort multiplier: `0.5`, `1.0`, `1.5` |

---

## Output Files

### `RESULTS.md`

Auto-generated markdown table aggregating key metrics across all completed experiments. Updated at the end of each `main.py` run. Contains per-scenario values of:

- **Attacker/Defender Reward** — cumulative reward averaged over test episodes
- **Goal Achievement (%)** — fraction of episodes where the attacker reached its goal
- **NVI (%)** — average percentage of compromised nodes
- **Mean TTRG (days)** — mean time to reach goal (attacker wins only)
- **Mean VIB** — mean vulnerabilities in the defender's backlog
- **Mean TTPV (days)** — mean time to complete a patch cycle

### `scenarios/*.json`

Raw per-episode data for each of the four scenarios. Each JSON file contains episode-level records used to generate all plots. These are the source of truth for any further analysis.

### Plot subdirectories

Each plot subdirectory corresponds to one metric and contains PNG files, one per scenario or combined across scenarios. Comparison plots under `comparisons/` overlay multiple experiments that share the same APT group and topology, enabling direct effort-level comparisons.
