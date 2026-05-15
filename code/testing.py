from NetworkExplorationEnv import NetworkExplorationEnv
from Scenario import Scenario
from PolicyDefender import PolicyDefender
from DQNAttacker import DQNAttacker
from logger import *

import simulation_parameters as sp

def test(dataset,topology):

    scenarios = {}

    for scenario_id, config in SCENARIOS.items():

        if not config.get("enabled", True):
            continue

        attacker = config["attacker"]
        defender = config["defender"]

        logger = scenario_loggers[scenario_id]

        if attacker == "DQN":
            scenario = scenario_dqn_attacker(
                dataset,
                scenario_id=scenario_id,
                logger=logger,
                defender_policy=defender,
                topology=topology
            )

        scenarios[scenario_id] = scenario

    return scenarios

def scenario_dqn_attacker(dataset, scenario_id, logger, topology, defender_policy=None):

    # Environment
    environment = NetworkExplorationEnv(
        num_nodes=sp.NUM_NODES,
        dataset_vulns=dataset,
        seed=sp.SEED_NETWORK,
        logger=logger,
        goal = sp.goal,
        topology=topology,
        phishing=sp.apt_phishing
    )

    # Attacker
    attacker = DQNAttacker(
        environment,
        logger=logger,
        seed=sp.SEED_NETWORK,
        name=sp.APT_NAME,
        goal=sp.goal,
        products=sp.apt_products,
        cve_list=sp.apt_vulns,
        phishing=sp.apt_phishing
    )

    # Defender (optional)
    defender = None
    scenario_name = f"Scenario {scenario_id}: DQN Attacker"

    if defender_policy is not None:
            defender = PolicyDefender(
                environment,
                defender_policy,
                logger=logger,
                seed=sp.SEED_NETWORK
            )
            scenario_name += f" - Policy Defender ({defender_policy.capitalize()})"
    else:
        scenario_name += " - No Defender"

    # Scenario
    scenario = Scenario(
        name=scenario_name,
        environment=environment,
        attacker=attacker,
        defender=defender,
        logger=logger,
        rl_attacker=True
    )

    # Run
    print(f"\nRunning {scenario_name}.")
    scenario.run_scenario(
        num_episodes=sp.TEST_EPISODES,
        seed=sp.SEED_TESTING
    )
    goals = sum(scenario.attacker_goal_achieved_per_episode)

    print(f"Completed {scenario_name} with {goals} goals achieved.\n")

    return scenario