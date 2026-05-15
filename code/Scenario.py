import numpy as np
import os
import torch
import json
import simulation_parameters as sp
from tqdm import tqdm

# Scenarios to simulate
SCENARIOS = {
    1: {"attacker": "DQN",   "defender": None,           "enabled": True},
    2: {"attacker": "DQN",   "defender": "importance",   "enabled": True},
    3: {"attacker": "DQN",   "defender": "severity",     "enabled": True},
    4: {"attacker": "DQN",   "defender": "centrality",   "enabled": True},
}

class Scenario :

    def __init__(self, name, environment,attacker,defender,logger,training=[False,False],rl_attacker=False):

        self.name = name
        self.environment = environment

        self.attacker = attacker
        self.defender = defender

        self.logger = logger
        self.training = training

        self.rl_attacker = rl_attacker

        if self.rl_attacker and not training[0]:
            self.attacker.model.load_state_dict(torch.load(f"{sp.MODEL_PATH}/dqn_attacker_trained.pth", map_location=self.attacker.device, weights_only=True))
            self.attacker.target_model.load_state_dict(self.attacker.model.state_dict())

        self.attacker_goal_achieved_per_episode = np.zeros(sp.TRAIN_EPISODES if training[0] else sp.TEST_EPISODES)

        self.rewards_attacker_per_episode = []
        self.rewards_defender_per_episode = []

        self.num_nodes_exploited_per_episode = []
        self.num_nodes_patched_per_episode = []

        self.num_vulnerabilities_exploited_per_episode = []
        self.num_vulnerabilities_patched_per_episode = []

        self.time_to_goal_ATTACKER_per_episode = []

        self.cumulative_reward_ATTACKER = []
        self.cumulative_reward_DEFENDER = []

        self.nvi_per_episode = []

        self.mean_vulns_in_backlog_per_episode = []
        self.mean_time_to_patch_per_episode = []

        self.attacker_total_visits_per_node = np.zeros(self.environment.num_nodes)
        self.attacker_mean_visits_per_node = np.zeros(self.environment.num_nodes)
        self.defender_total_visits_per_node = np.zeros(self.environment.num_nodes)
        self.defender_mean_visits_per_node = np.zeros(self.environment.num_nodes)

        self.total_vulns_per_node = np.zeros(self.environment.num_nodes)
        self.mean_vulns_per_node = np.zeros(self.environment.num_nodes)

        self.attacker_mean_node_visits = np.zeros(self.environment.num_nodes)
        self.defender_mean_node_visits = np.zeros(self.environment.num_nodes)

    
    def to_dict(self):
        return {
            "name": self.name,
            "training": self.training,
            "rl_attacker": self.rl_attacker,

            "attacker_goal_achieved_per_episode": self.attacker_goal_achieved_per_episode.tolist(),

            "rewards_attacker_per_episode": self.rewards_attacker_per_episode,
            "rewards_defender_per_episode": self.rewards_defender_per_episode,

            "num_nodes_exploited_per_episode": self.num_nodes_exploited_per_episode,
            "num_nodes_patched_per_episode": self.num_nodes_patched_per_episode,

            "num_vulnerabilities_exploited_per_episode": self.num_vulnerabilities_exploited_per_episode,
            "num_vulnerabilities_patched_per_episode": self.num_vulnerabilities_patched_per_episode,

            "time_to_goal_ATTACKER_per_episode": self.time_to_goal_ATTACKER_per_episode,

            "cumulative_reward_ATTACKER": self.cumulative_reward_ATTACKER,
            "cumulative_reward_DEFENDER": self.cumulative_reward_DEFENDER,

            "nvi_per_episode": self.nvi_per_episode,
            "mean_vulns_in_backlog_per_episode": self.mean_vulns_in_backlog_per_episode,
            "mean_time_to_patch_per_episode": self.mean_time_to_patch_per_episode,

            "attacker_total_visits_per_node": self.attacker_total_visits_per_node.tolist(),
            "attacker_mean_visits_per_node": self.attacker_mean_visits_per_node.tolist(),

            "defender_total_visits_per_node": self.defender_total_visits_per_node.tolist(),
            "defender_mean_visits_per_node": self.defender_mean_visits_per_node.tolist(),

            "total_vulns_per_node": self.total_vulns_per_node.tolist(),
            "mean_vulns_per_node": self.mean_vulns_per_node.tolist(),

            "attacker_mean_node_visits": self.attacker_mean_node_visits.tolist(),
            "defender_mean_node_visits": self.defender_mean_node_visits.tolist()
        }

    def save_to_json(self, path):
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            return obj

        path = os.path.normpath(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4, default=convert)

    def run_scenario(self,num_episodes,seed):
        '''
        Run the scenario with the attacker and defender agents.
        '''
        if self.defender is None:
            self.run_scenario_Attacker_ONLY(num_episodes,seed)
        else:
            self.run_scenario_Attacker_VS_Defender(num_episodes,seed)
        
        self.save_to_json(f"{sp.exp_dir}/scenarios/{self.name.replace(' ', '_')}.json")


    def run_training(self,num_episodes,seed):
        '''
        Train the agents in the given environment.
        '''
        if self.rl_attacker:
            self.run_training_DQN_Attacker(num_episodes,seed)


    def run_training_DQN_Attacker(self, num_episodes, seed):
        for episode in tqdm(range(num_episodes)):
            total_A, total_D = self._run_episode(episode, seed, training=True)
            self._update_metrics(episode, total_A)

            seed += 1

        self.cumulative_reward_ATTACKER = np.cumsum(self.rewards_attacker_per_episode)

        os.makedirs(sp.MODEL_PATH, exist_ok=True)
        torch.save(self.attacker.model.state_dict(), f"{sp.MODEL_PATH}/dqn_attacker_trained.pth")

    def run_scenario_Attacker_ONLY(self, num_episodes, seed):
        for episode in tqdm(range(num_episodes)):
            total_A, _ = self._run_episode(episode, seed)
            self._update_metrics(episode, total_A)

            seed += 1

        self.cumulative_reward_ATTACKER = np.cumsum(self.rewards_attacker_per_episode)
        self._finalize_statistics(num_episodes)


    def run_scenario_Attacker_VS_Defender(self, num_episodes, seed):
        for episode in tqdm(range(num_episodes)):
            total_A, total_D = self._run_episode(episode, seed, with_defender=True)
            self._update_metrics(episode, total_A, total_D, with_defender=True)

            seed += 1

        self.cumulative_reward_ATTACKER = np.cumsum(self.rewards_attacker_per_episode)
        self.cumulative_reward_DEFENDER = np.cumsum(self.rewards_defender_per_episode)
        self._finalize_statistics(num_episodes, with_defender=True)

    def _run_episode(self, episode, seed, with_defender=False, training=False):
        self.environment.reset(seed)

        self.attacker.reset(seed)
        if with_defender:
            self.defender.reset(seed)

        state = self.environment.get_state() if training else None

        total_reward_A = 0
        total_reward_D = 0
        done = False
        day = 0

        while (not done) if training else (day <= sp.TIME_LIMIT):
            day += 1
            self.logger.debug(f"[SCENARIO][EPISODE {episode+1}] Time: {day}")

            attacker_action = self.attacker.step()

            if attacker_action is not None:
                self.attacker_total_visits_per_node[attacker_action[0]] += 1

            defender_action = None
            if with_defender:
                defender_action = self.defender.step()
                if defender_action:
                    for a in defender_action:
                        self.defender_total_visits_per_node[a[0]] += 1

            next_states, rewards, dones = self.environment.step(attacker_action, defender_action)

            if training and attacker_action is not None:
                self.attacker.add_to_memory(
                    state[0], attacker_action[0], attacker_action[1],
                    rewards[0], next_states[0], dones[0]
                )
                state = next_states
                self.attacker.decay_epsilon()

            if dones[0]:
                self.attacker_goal_achieved_per_episode[episode] = 1

            total_reward_A += rewards[0]
            if with_defender:
                total_reward_D += rewards[1]

            done = dones[0] or dones[1]

        return total_reward_A, total_reward_D

    def _update_metrics(self, episode, total_A, total_D=0, with_defender=False):
        self.rewards_attacker_per_episode.append(total_A)

        if with_defender:
            self.rewards_defender_per_episode.append(total_D)

        self.num_nodes_exploited_per_episode.append(self.environment.count_exploited_nodes())
        self.num_vulnerabilities_exploited_per_episode.append(self.environment.count_exploited_vulnerabilities())

        if with_defender:
            self.num_nodes_patched_per_episode.append(self.environment.count_patched_nodes())
            self.num_vulnerabilities_patched_per_episode.append(self.environment.count_patched_vulnerabilities())
            self.mean_vulns_in_backlog_per_episode.append(np.mean(self.defender.vulns_in_backlog))
            self.mean_time_to_patch_per_episode.append(np.mean(self.environment.time_to_patch))

        self.nvi_per_episode.append(self.environment.calculate_NVI())

        if self.environment.time_to_goal_ATTACKER is not None:
            self.time_to_goal_ATTACKER_per_episode.append(self.environment.time_to_goal_ATTACKER)
    
    def _finalize_statistics(self, num_episodes, with_defender=False):
        self.attacker_mean_node_visits = self.attacker_total_visits_per_node / num_episodes

        if with_defender:
            self.defender_mean_node_visits = self.defender_total_visits_per_node / num_episodes

        self.mean_vulns_per_node = self.total_vulns_per_node / num_episodes