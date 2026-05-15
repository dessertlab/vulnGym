import Vulnerability
import training, testing
import plots
import results
import simulation_parameters as sp
import logger as logger
import os

all_experiments = {}
current_experiment = None

for experiment_id in range(1, sp.NUM_EXPERIMENTS + 1):

    sp.load_experiment(experiment_id)
    logger.setup_experiment_loggers(experiment_id)

    if sp.EXPERIMENT != current_experiment:
        DATASET_VULNERABILITY = Vulnerability.load_vulnerabilities_from_dataset(sp.DATASET_PATH)
        current_experiment = sp.EXPERIMENT

    if not os.path.exists(f"{sp.MODEL_PATH}/dqn_attacker_trained.pth"):
        print(f"No trained model found for {sp.APT_NAME} on {sp.TOPOLOGY}. Training...")
        training.train(DATASET_VULNERABILITY, sp.TOPOLOGY)
    else:
        print(f"Trained model found for {sp.APT_NAME} on {sp.TOPOLOGY}. Skipping training.")

    print(f"Testing experiment {experiment_id}.")

    scenarios = testing.test(DATASET_VULNERABILITY, sp.TOPOLOGY)
    plots.plot_all(scenarios, sp.exp_dir)

    all_experiments[experiment_id] = {
        "scenarios": scenarios,
        "apt_name": sp.APT_NAME,
        "topology": sp.TOPOLOGY,
        "defender_effort": sp.DEFENDER_EFFORT,
        "exp_dir": sp.exp_dir,
    }

    print(f"Plots for experiment {experiment_id} saved in {sp.exp_dir}.\n")

results.update_results_md(all_experiments)
