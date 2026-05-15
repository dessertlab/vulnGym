# VulnGym

A simulation tool that quantitatively evaluates vulnerability management strategies by capturing the dynamics of complex attack campaigns.

---

## Overview

VulnGym models the interaction between:

- **Attacker** — a Deep Q-Network (DQN) agent trained to compromise a target network, parameterized by a real APT group's toolset, targeted products, and attack goal.
- **Defender** — a policy-based agent that discovers and patches vulnerabilities according to a configurable prioritization strategy and effort level.

The environment is a network of nodes organized into security zones, populated with real CVE data from NVD.

---

## Project Structure

```
vulnGym/
├── code/
│   ├── main.py                     # Entry point — runs all experiments
│   ├── training.py                 # DQN training loop
│   ├── testing.py                  # Evaluation loop
│   ├── DQNAttacker.py              # DQN agent implementation
│   ├── NetworkExplorationEnv.py    # Gymnasium environment
│   ├── NetworkNode.py              # Network node model
│   ├── PolicyDefender.py           # Rule-based defender agent
│   ├── Scenario.py                 # Scenario orchestration
│   ├── Vulnerability.py            # Vulnerability data model
│   ├── plots.py                    # Plot generation
│   ├── results.py                  # Results aggregation and RESULTS.md writer
│   ├── rewards.py                  # Reward function definitions
│   ├── simulation_parameters.py    # Global hyperparameters
│   ├── logger.py                   # Logging utilities
│   ├── macro.py                    # Shared macros / constants
│   ├── clear.py                    # Utility to clean results and logs
│   ├── viewer.py                   # Viewer helpers for the notebook
│   └── experiment_viewer.ipynb     # Interactive results browser
├── config/
│   ├── apt_groups.json             # APT group profiles
│   └── products.json               # Product-to-zone mapping
├── datasets/
│   ├── nvd/                        # NVD records (2020–2024)
│   ├── kev/                        # CISA KEV catalog (2020–2024)
│   └── metasploit-nuclei/          # Metasploit/Nuclei CVE lists (2020–2024)
├── trained_models/
│   ├── layered/
│   │   ├── APT28/                  # Saved DQN models — layered topology, APT28
│   │   └── APT41/                  # Saved DQN models — layered topology, APT41
│   └── tree/
│       ├── APT28/                  # Saved DQN models — tree topology, APT28
│       └── APT41/                  # Saved DQN models — tree topology, APT41
├── results/                        # Auto-generated output (see Results section)
├── logs/                           # Per-experiment episode logs
├── experimental_plan.xlsx          # Experiment definitions
└── requirements.txt
```

---

## Installation

It is recommended to create a Python virtual environment before installing dependencies:

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Dependencies:** `gymnasium`, `torch`, `numpy`, `networkx`, `matplotlib`, `scipy`, `tqdm`, `pandas`, `openpyxl`

---

## Quickstart

Run `code/main.py` to execute all experiments defined in `experimental_plan.xlsx`:

```bash
python code/main.py
```

For each experiment it:

1. Loads the configuration from `experimental_plan.xlsx` by `Experiment_ID`.
2. Loads the vulnerability dataset (shared across experiments with the same APT and topology).
3. Trains the DQN attacker if no saved model is found; otherwise skips training.
4. Runs the evaluation scenarios (attacker-only and attacker vs. each defender policy).
5. Generates and saves all plots to `results/<experiment_dir>/`.
6. Writes a summary of all experiments to `results/results.md`.

To visualize results interactively, open `code/experiment_viewer.ipynb`:

- **Section 1** — set `EXPERIMENT_ID` and call `display_experiment(EXPERIMENT_ID)` to browse all saved plots for a single experiment.
- **Section 2** — set `EXPERIMENT_IDS` (a list of IDs sharing the same APT and topology) and call `display_comparison(EXPERIMENT_IDS)` to generate and display cross-experiment comparison plots.

---

## Experiment Configuration

Each experiment is defined as a row in `experimental_plan.xlsx` with the following columns:

| Column | Description |
| --- | --- |
| `Experiment_ID` | Unique integer identifier |
| `APT_Name` | Name of the APT group (must match an entry in `config/apt_groups.json`) |
| `Campaign` | Attack campaign type: `impact` or `dos` |
| `Year` | Dataset year: 2020–2024 |
| `Network` | Topology: `layered` or `tree` |
| `Network_Size` | Total number of nodes in the simulated network |
| `Defender_Effort` | Defender effort multiplier (affects patching time) |

---

## Network Model

The simulated network is divided into four security zones, each with a different node importance range:

| Zone | Role | Importance |
| --- | --- | --- |
| `external` | Internet-facing nodes | Low (1-3) |
| `dmz` | Demilitarized zone (web servers, proxies) | Medium (3-6) |
| `internal` | Corporate network (workstations, services) | Medium-High (6-9) |
| `database` | Database servers (primary attack target) | High (9-10) |

### Topologies

- **`layered`** — Zones are fully connected internally; cross-zone connections go through a fixed set of shared gateway nodes.
- **`tree`** — Zones are split into subnets; only a fraction of nodes act as gateways to the next zone (sparse cross-zone connections).

Vulnerabilities from the NVD dataset are assigned to nodes based on product match. Each node runs 3 products selected from `config/products.json`, which specifies valid deployment zones per product. Vulnerabilities are revealed progressively during the simulation based on their NVD publication date.

---

## Agents

### DQN Attacker

A DQN agent that learns to exploit vulnerabilities across the network to reach a campaign goal. The attacker is parameterized by an APT group profile:

- **Supported products** — determines which nodes are attackable
- **CVE list** — KEV + Metasploit/Nuclei CVEs for the selected year; affects exploit success probability
- **Phishing** — if enabled, the attacker can directly reach internal nodes without pivoting
- **Goal** — `data_exfiltration` (compromise ≥3 database nodes via exfiltration/wiper) or `db_denial_of_service` (take down ≥3 database nodes via DoS)

### Policy Defender

A rule-based defender that runs periodic discovery-and-patch cycles. Each cycle consists of:

1. **Discovery phase** — scans all online nodes and registers unpatched vulnerabilities.
2. **Resolution phase** — selects one vulnerability to patch based on the configured policy and applies it over a calculated number of days.

**Patch prioritization policies:**

| Policy | Criterion |
| --- | --- |
| `importance` | Average node importance of affected nodes (highest first) |
| `severity` | CVSS score of the vulnerability (highest first) |
| `centrality` | Average betweenness centrality of affected nodes (highest first) |
| `random` | Random selection |

**Patching time** is a function of the defender effort multiplier (`DEFENDER_EFFORT`), the product type (application vs. OS), and the number of affected nodes.

---

## Scenarios

Four scenarios are evaluated for each experiment:

| Scenario | Attacker | Defender |
| --- | --- | --- |
| 1 | DQN | None (unconstrained attacker) |
| 2 | DQN | Importance policy |
| 3 | DQN | Severity policy |
| 4 | DQN | Centrality policy |

Each scenario runs for `TEST_EPISODES = 100` episodes of up to `TIME_LIMIT = 365` simulated days. Results are saved as JSON files under `results/<experiment_dir>/scenarios/`.

---

## APT Configuration

APT groups are defined in `config/apt_groups.json`. Each entry has:

```json
{
  "name": "APT41",
  "mitre_url": "https://attack.mitre.org/groups/G0096/",
  "products": ["apache:apache", "microsoft:windows", ...],
  "phishing": true,
  "goal": "data_exfiltration"
}
```

To add a new APT group, append an entry to `apt_groups.json` and add a corresponding row to `experimental_plan.xlsx`.

---

## Key Metrics

Results are reported per episode and averaged over the test set:

| Metric | Description |
| --- | --- |
| **Goal Achievement Rate** | Fraction of episodes where the attacker reached the goal |
| **NVI (Network Vulnerability Index)** | Percentage of nodes compromised |
| **TTRG (Time to Reach Goal)** | Days elapsed when the attacker wins |
| **VIB (Vulnerabilities in Backlog)** | Unpatched vulns queued for the defender |
| **TTPV (Time to Patch Vulnerability)** | Days to complete a patch cycle |

---

## Simulation Parameters

Key parameters are set in `code/simulation_parameters.py`:

| Parameter | Default | Description |
| --- | --- | --- |
| `TIME_LIMIT` | 365 | Max simulation steps per episode (days) |
| `TRAIN_EPISODES` | 1000 | DQN training episodes |
| `TEST_EPISODES` | 100 | Evaluation episodes |
| `MAX_VULNS_PER_NODE` | 10 | Max vulnerabilities assigned to a single node |
| `NUM_PRODUCTS_PER_NODE` | 3 | Products installed per node |
| `TARGET_GOALS` | 3 | Database nodes the attacker must compromise to win |
| `SUCCESS_PROBABILITY_ATTACKER` | 1.0 | Base exploit success probability |
| `SUCCESS_PROBABILITY_DEFENDER` | 1.0 | Base patch success probability |

---

## Results

Outputs are written to `results/` after each run. The structure of the folder and the summary table are documented in [`results/README.md`](results/README.md).

### Folder layout

```
results/
├── RESULTS.md                                              # Auto-generated summary table
├── comparisons/exp_<N1>__<N2>__<N3>/                      # Cross-experiment comparison plots
└── exp_<NN>__<APT>__<Campaign>__<Topology>__effort_<X>/   # One folder per experiment
    ├── network_<topology>.png       # Network topology visualization
    ├── scenarios/                   # Raw JSON data for each of the 4 scenarios
    ├── heatmaps/                    # Node-level exploitation and patching heatmaps
    ├── rewards/                     # Reward curves per episode
    ├── goal/                        # Goal achievement rate
    ├── nvi/                         # Network Vulnerability Index over time
    ├── nodes/                       # Nodes exploited vs. patched
    ├── vulns/                       # Vulnerabilities exploited vs. patched
    ├── backlog/                     # Defender vulnerability backlog
    └── time/                        # Time-to-goal and time-to-patch
```

### What is generated

- **`RESULTS.md`** — aggregated metrics table updated at the end of every `main.py` run.
- **`scenarios/*.json`** — raw episode-level data for each scenario; source of truth for all plots.
- **Plot subdirectories** — one PNG per metric per experiment. Comparison plots under `comparisons/` overlay experiments sharing the same APT group and topology to allow direct effort-level comparison.

---

## Datasets

- **`datasets/nvd/`** — NVD records with CVE IDs, CVSS scores, CWE classifications, affected products, and publication dates. Used to populate node vulnerabilities.
- **`datasets/kev/`** — CISA KEV catalog. CVEs in this list grant the attacker a higher exploit success probability.
- **`datasets/metasploit-nuclei/`** — CVEs covered by Metasploit modules or Nuclei templates. Combined with KEV to define the attacker's effective exploit arsenal.
