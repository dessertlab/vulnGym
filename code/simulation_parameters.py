import glob
import json
import os
import warnings
import pandas as pd
# Root of the project (one level above this file)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Suppress warnings from tqdm
warnings.filterwarnings("ignore", category=Warning, module="tqdm")

# =========================
# Fixed parameters
# =========================

# Maximum number of steps per episode (simulation time horizon)
TIME_LIMIT = 365

# Base success probability of the attacker (1.0 = always succeeds if conditions are met)
SUCCESS_PROBABILITY_ATTACKER = 1.0

# Base success probability of the defender (1.0 = patching always succeeds)
SUCCESS_PROBABILITY_DEFENDER = 1.0

# =========================
# Episodes
# =========================

# Number of episodes used during testing (model evaluation phase)
TEST_EPISODES = 100

# Number of episodes used for training the DQN agent
TRAIN_EPISODES = 1000

# Number of recent episodes to track
NUM_EPISODES_TO_TRACK = 10

# =========================
# Seeds (reproducibility)
# =========================

# Seed for network topology generation
SEED_NETWORK = 12345

# Seed for vulnerability generation/assignment
SEED_VULNERABILITIES = 42

# Seed for assigning products to nodes
SEED_PRODUCTS = 42

# =========================
# Simulation seeds
# =========================

# Seed used during training (ensures reproducible training runs)
SEED_TRAINING = 13345

# Seed used during testing (different from training to avoid overfitting)
SEED_TESTING = 56789


# =========================
# Derived parameters
# =========================

# Maximum number of vulnerabilities that can be assigned to a single node
MAX_VULNS_PER_NODE = 10

# Number of products/services installed on each node
NUM_PRODUCTS_PER_NODE = 3

# Number of target goals the attacker must achieve (e.g., compromise/exfiltrate from N nodes)
TARGET_GOALS = 3

# Load the experimental plan from Excel
df_plan = pd.read_excel(os.path.join(_ROOT, "experimental_plan.xlsx"))
NUM_EXPERIMENTS = df_plan["Experiment_ID"].max()

def find_exp_dir(row: pd.Series) -> str:
    folder_name = (
        f"exp_{int(row.Experiment_ID):02d}__"
        f"{row.APT_Name}__{row.Campaign}__{row.Network}__"
        f"effort_{row.Defender_Effort}"
    )
    path = os.path.join(_ROOT, "results", folder_name)
    os.makedirs(path, exist_ok=True)
    return path


def load_experiment(experiment_id: int):
    """
    Load a full experiment configuration from Excel.

    This function:
    - reads the experimental plan
    - selects the row by experiment_id
    - initializes all simulation parameters
    - computes derived variables
    """

    # =========================
    # Experiment setup
    # =========================

    # Select the row corresponding to the given experiment ID
    row = df_plan[df_plan["Experiment_ID"] == experiment_id]

    # Raise an error if the experiment ID is not found
    if row.empty:
        raise ValueError(f"Experiment {experiment_id} not found.")

    # Extract the first matching row
    row = row.iloc[0]

    # =========================
    # Base parameters (from Excel)
    # =========================
    global YEAR, APT_NAME, CAMPAIGN, TOPOLOGY, NUM_NODES, DEFENDER_EFFORT

    # Year of the dataset
    YEAR = int(row["Year"])
    # Name of the APT group
    APT_NAME = row["APT_Name"]
    # Campaign type
    CAMPAIGN = row["Campaign"].lower()
    # Network topology type
    TOPOLOGY = row["Network"].lower()
    # Number of nodes in the simulated network
    NUM_NODES = int(row["Network_Size"])

    # Defender effort parameter (affects patching time or effectiveness)
    DEFENDER_EFFORT = float(row["Defender_Effort"])

    # =========================
    # Extra experiment info
    # =========================
    global exp_dir

    # Compute the directory where experiment results will be stored
    exp_dir = find_exp_dir(row)

    # Path to the trained model for this (APT, topology) pair
    global MODEL_PATH
    MODEL_PATH = os.path.join(_ROOT, "trained_models", TOPOLOGY, APT_NAME)
    #print(f"Running: {exp_dir}")

    # =========================
    # Dataset loading
    # =========================
    global DATASET_PATH
    DATASET_PATH = os.path.join(_ROOT, "datasets", "nvd", f"nvd_{YEAR}.json")

    global EXPERIMENT
    EXPERIMENT = f"{APT_NAME}_{TOPOLOGY}_{YEAR}"

    # =========================
    # Load APT
    # =========================
    global goal, apt_products, apt_database_products, apt_phishing, apt_vulns

    # Load APT group definitions from JSON file
    with open(os.path.join(_ROOT, "config", "apt_groups.json"), "r", encoding="utf-8") as f:
        chosen_apt = next((a for a in json.load(f) if a["name"] == APT_NAME), None)

    with open(os.path.join(_ROOT, "config", "products.json"), "r", encoding="utf-8") as f:
        products_data = json.load(f)

    database_products = [product for product, info in products_data.items() if "database" in info.get("zones", [])]

    # Raise an error if the APT is not found
    if chosen_apt is None:
        raise ValueError(f"APT {APT_NAME} not found.")

    # Extract attacker goal (e.g., exfiltration, DoS, etc.)
    goal = chosen_apt.get("goal")

    # List of products targeted/supported by the APT
    apt_products = chosen_apt.get("products", [])

    # Whether the APT can perform phishing attacks
    apt_phishing = chosen_apt.get("phishing", False)

    kev_vulns = load_cve_list(os.path.join(_ROOT, "datasets", "kev"), YEAR)

    metasploit_vulns = load_cve_list(os.path.join(_ROOT, "datasets", "metasploit-nuclei"), YEAR)

    # Load the CVE dataset corresponding to the selected year
    apt_vulns = set(kev_vulns + metasploit_vulns)

    # List of database products
    apt_database_products = [p for p in apt_products if p in database_products]


def load_cve_list(directory, year):
    """
    Load the CVE dataset file corresponding to a given year.

    The function searches for JSON files in the specified directory
    matching the pattern '*{year}*.json', selects the first match
    (after sorting), and returns its content as a Python object.

    Args:
        directory (str): Path to the directory containing CVE dataset files.
        year (int or str): Target year used to filter the dataset file.

    Returns:
        dict or list: Parsed JSON content of the selected CVE file.

    Raises:
        FileNotFoundError: If no file matching the given year is found.
    """

    # Build the file search pattern
    pattern = os.path.join(directory, f"*{year}*.json")

    # Retrieve and sort all matching files
    files = sorted(glob.glob(pattern))

    # If no matching file is found, raise an error
    if not files:
        raise FileNotFoundError(f"No file found for year {year} in {directory}")

    # Open the first matching file and load its JSON content
    with open(files[0], "r", encoding="utf-8") as f:
        return json.load(f)