from NetworkExplorationEnv import NetworkExplorationEnv
from DQNAttacker import DQNAttacker
import Scenario
import simulation_parameters as sp

from logger import logger_training_DQNAttacker

def train(dataset, topology):

    # Train Attacker
    train_attacker(dataset, topology)


def train_attacker(dataset, topology):
    # Environment Definition
    environment_training_attacker = NetworkExplorationEnv(num_nodes=sp.NUM_NODES, dataset_vulns=dataset, seed=sp.SEED_NETWORK, logger=logger_training_DQNAttacker, goal=sp.goal, topology=topology, phishing=sp.apt_phishing)

    # Scenario Definition
    scenario_training_attacker = Scenario.Scenario(
        name = "Training Scenario: DQN Attacker - Policy Defender",
        environment= environment_training_attacker,
        attacker = DQNAttacker(environment_training_attacker, logger=logger_training_DQNAttacker, seed=sp.SEED_NETWORK,name=sp.APT_NAME, goal=sp.goal, products=sp.apt_products, cve_list=sp.apt_vulns, phishing=sp.apt_phishing, training = True),
        defender = None,
        logger = logger_training_DQNAttacker,
        training = [True,False] , rl_attacker=True)

    # Start Training

    print("Starting training for DQN Attacker...")

    scenario_training_attacker.run_training(
        num_episodes=sp.TRAIN_EPISODES,
        seed=sp.SEED_TRAINING
    )

    goals = sum(scenario_training_attacker.attacker_goal_achieved_per_episode)
  
    print(f"Training for DQN Attacker completed with {goals} goals achieved.\n")