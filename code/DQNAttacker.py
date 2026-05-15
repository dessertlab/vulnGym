import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import simulation_parameters as sp

from collections import deque
from macro import *

# Hyperparameters
LEARNING_RATE   = 0.001
DISCOUNT_FACTOR = 0.99
EPSILON         = 1.0
EPSILON_DECAY   = 0.995
EPSILON_MIN     = 0.01
MEMORY_SIZE     = 10000
BATCH_SIZE      = 64
TARGET_UPDATE   = 1000

NUM_NODE_ACTIONS = len(ACTION_TYPES)

class DQNNetwork(nn.Module):
    """Single-head Q-network: Q(s, (node, action)) — output shape [num_nodes * 9]."""

    def __init__(self, state_size, action_size):
        super().__init__()
        self.out_size = action_size
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, self.out_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q = self.out(x)   # [B, num_nodes * 9]
        return q


class DQNAttacker:
    def __init__(self, env, logger, seed, name, goal, products, cve_list, phishing=False, success_rate=sp.SUCCESS_PROBABILITY_ATTACKER, training=False):
        random.seed(seed)

        self.logger = logger
        self.env = env

        self.success_rate = success_rate

        self.k = 3
        self.attack_time = 0
        # Phase flag: -2 = start (no pivot yet), -1 = pivot selection, >0 = preparing, 0 = execute
        self.attack_time_remaining = -2
        self.chosen_node = None
        self.actual_node = None
        self.chosen_action = WAIT
        self.chosen_vuln = None

        # RL training parameters
        self.alpha = LEARNING_RATE
        self.gamma = DISCOUNT_FACTOR
        self.epsilon = EPSILON
        self.epsilon_decay = EPSILON_DECAY
        self.epsilon_min = EPSILON_MIN
        self.batch_size = BATCH_SIZE

        self.memory = deque(maxlen=MEMORY_SIZE)
        self.loss_history = deque(maxlen=200)
        self.train_counter = 0
        self.train_interval = 8
        self.training = training

        self.failed_visited_nodes = []
        self.failed_starting_nodes = set()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_size = self.env.num_nodes * NUM_NODE_ACTIONS

        self.state_size = env.state_size_attacker

        self.model = DQNNetwork(self.state_size, self.action_size).to(self.device)
        self.target_model = DQNNetwork(self.state_size, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        self.target_update_freq = TARGET_UPDATE
        self.update_target_network()
        self.step_counter = 0

        self.name = name
        self.goal = goal
        self.products = list(products) if products is not None else []
        self.cve_list = list(cve_list) if cve_list is not None else []
        self.phishing = phishing


    # ------------------------------------------------------------------ #
    # Reset & infrastructure
    # ------------------------------------------------------------------ #
    def reset(self, seed):
        random.seed(seed)
        self.attack_time = 0
        self.attack_time_remaining = -2
        self.chosen_node = None
        self.actual_node = None
        self.chosen_action = WAIT
        self.chosen_vuln = None
        self.train_counter = 0
        self.failed_visited_nodes.clear()
        self.failed_starting_nodes.clear()
        self.logger.debug("[DQN][RESET] State and memory reset")

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    # ------------------------------------------------------------------ #
    # Main loop: step → select → wait → execute
    # ------------------------------------------------------------------ #
    def step(self):
        """Always returns (node, action, vuln, success) or None while waiting/selecting."""
        self.attack_time += 1

        # Wait until at least one vulnerable node is visible in the early simulation ticks
        if self.env.current_time < 50 and not any(
            obj.state == NODE_DANGER for obj in self.env.node_objects.values()
        ):
            self.logger.debug("[DQN][STEP] No vulnerable node visible. WAIT.")
            action = None
            return action

        # Reset phase when the current pivot node changes
        if self.actual_node != self.env.current_node:
            self.attack_time = 0
            self.chosen_node = None
            self.chosen_action = WAIT
            self.chosen_vuln = None
            self.actual_node = self.env.current_node
            # If the attacker has a pivot node, move to pivot-selection; otherwise back to start
            self.attack_time_remaining = -2 if self.env.current_node is None else -1

        if self.env.current_node is None and self.attack_time_remaining == -1:
            self.attack_time_remaining = -2

        # Select (node, action, vuln) triple during start (-2) or pivot (-1) phases
        if self.attack_time_remaining in (-2, -1):
            self.chosen_node = self._select_target_node()
            action = None
            return action

        # Wait while the attacker prepares (preparation timer counts down)
        if self.attack_time_remaining > 0:
            self.attack_time_remaining -= 1
            action = None
            return action

        # Execute the chosen action once the preparation timer reaches zero
        if self.attack_time_remaining == 0:
            self.chosen_action, success = self._execute_action_for_node(self.chosen_node)
            action = [self.chosen_node, self.chosen_action, self.chosen_vuln, success]
            return action

    # ------------------------------------------------------------------ #
    # Node, action, and vulnerability selection
    # ------------------------------------------------------------------ #
    def _select_target_node(self):

        def get_exploitable_vulns(o):
            return [
                v for v in getattr(o, "vulnerabilities", [])
                if not getattr(v, "patched", False)
                and getattr(v, "exploitable", None) is not False
            ]

        def is_valid_node(o):
            return (
                o.state == NODE_DANGER
                and not o.is_off
                and getattr(o, "attackable", None) is not False
                and get_exploitable_vulns(o)
            )

        def valid_nodes(nodes):
            return [
                n for n in nodes
                if is_valid_node(self.env.network.nodes[n]["node"])
            ]

        neighbors_from_credentials = []

        # --- START phase (-2): pick from initial entry points ---
        if self.attack_time_remaining == -2:

            if self.phishing:
                external = [n for n, o in self.env.node_objects.items() if o.zone == "external"]
                internal = [n for n, o in self.env.node_objects.items() if o.zone == "internal" and o.phishing]
                base_nodes = set(external) | set(internal)
            else:
                base_nodes = set(self.env.starting_nodes)

            candidates = valid_nodes(base_nodes)

        # --- PIVOT phase (-1): expand from already-compromised nodes ---
        elif self.attack_time_remaining == -1:

            compromised = {n for n, o in self.env.node_objects.items() if o.attacker_has_access}

            neighbors = {
                neigh
                for c in compromised
                for neigh in self.env.network.neighbors(c)
            }

            credentials_nodes = {n for n, o in self.env.node_objects.items() if o.credentials_stolen}
            neighbors_from_credentials = {
                neigh
                for c in credentials_nodes
                for neigh in self.env.network.neighbors(c)
                if neigh not in compromised
            }

            if self.phishing:
                external = [n for n, o in self.env.node_objects.items() if o.zone == "external"]
                internal = [n for n, o in self.env.node_objects.items() if o.zone == "internal" and o.phishing]
                phishing_nodes = set(valid_nodes(set(external) | set(internal)))

                candidates = set(valid_nodes(neighbors)) | compromised | phishing_nodes | neighbors_from_credentials

            else:
                starting = set(valid_nodes(self.env.starting_nodes))
                candidates = compromised | set(valid_nodes(neighbors)) | starting | neighbors_from_credentials

            candidates = list(candidates)

        # Build (node, action, vuln) candidate triples
        pairs = []

        for n in candidates:
            o = self.env.network.nodes[n]["node"]
            vulns = get_exploitable_vulns(o)

            if o.attacker_has_access:
                exploited = [v for v in vulns if getattr(v, "exploited", False)]
                pairs.extend(self.env._valid_post_actions_for_node(n, exploited))

            else:
                if not getattr(o, "scanned", False):
                    pairs.append((n, SCAN, None))

                if n in neighbors_from_credentials:
                    pairs.append((n, LATERAL_MOVEMENT, None))

                for v in vulns:
                    if REMOTE_CONTROL in v.vulnerability_type:
                        pairs.append((n, EXPLOIT, v))
                    if DOS in v.vulnerability_type:
                        pairs.append((n, DOS, v))

        if not pairs:
            self.logger.debug("[DQN] No valid triple found. WAIT.")
            return None

        state = self.env.get_state()
        node, action, vuln = self.choose_action_dqn(state[0], pairs)

        self.chosen_node = node
        self.chosen_action = action
        self.chosen_vuln = vuln

        if node is not None:
            self.update_attack_time(vuln)
            self.attack_time = 0

        return node

    # ------------------------------------------------------------------ #
    # Execute the decision for the chosen node
    # ------------------------------------------------------------------ #
    def _execute_action_for_node(self, target_node):
        """
        Uses the (node, action, vuln) triple chosen in _select_target_node.
        - DOS: calls perform_action(node, DOS, vuln)
        - Uncompromised node: SCAN → perform_remote_scan; EXPLOIT → probability-based
        - Compromised node: dispatches the chosen post-exploitation action
        """
        self.attack_time_remaining = -1
        success = False

        if target_node is None:
            return WAIT, False

        o = self.env.network.nodes[target_node]["node"]

        # Vulnerability selected during the selection phase
        vuln = self.chosen_vuln

        # --- DoS attack ---
        if self.chosen_action == DOS:

            if (o.state != NODE_DANGER or vuln.patched):
                self.logger.debug(f"[DQN][{self.chosen_action}] Node {target_node} not DANGER.")
            if not any(p in self.products for p in o.products):
                self.logger.debug(f"[DQN][{self.chosen_action}] Node {target_node} no products '{o.products}' supported.")
                o.attackable = False
            if vuln.cve_id not in self.cve_list:
                self.logger.debug(f"[DQN][{self.chosen_action}] '{vuln.cve_id}' not in KEV/Metasploit list.")
                vuln.exploitable = False

            prob = self.calculate_success_probability(target_node, self.chosen_action, self.chosen_vuln)
            success = (np.random.rand() < prob)

            if (success):
                self.logger.debug(f"[DQN][{self.chosen_action}] Node {target_node} - execute {self.chosen_action}.")
            else:
                self.logger.debug(f"[DQN][{self.chosen_action}] Node {target_node} - failed {self.chosen_action}.")
                if self.env.current_node is None:
                    self.failed_starting_nodes.add(target_node)
                    self.attack_time_remaining = -2
                else:
                    self.attack_time_remaining = -1
                    if target_node not in self.failed_visited_nodes:
                        self.failed_visited_nodes.append(target_node)

            return DOS, success

        # --- Uncompromised node ---

        # SCAN: discover products and vulnerabilities on the target
        if self.chosen_action == SCAN and not getattr(o, "scanned", False):
            self.env.perform_remote_scan(target_node, supported_products=self.products,
                                                        supported_vulns=self.cve_list)

            self.logger.debug(
                f"[DQN][_execute_action_for_node] SCAN on node={target_node}, "
                f" attackable={o.attackable}"
                f" product={getattr(o, 'product', None)}."
            )
            return SCAN, True

        # EXPLOIT: attempt remote code execution via a known vulnerability
        if self.chosen_action == EXPLOIT:

            if (o.state != NODE_DANGER or vuln.patched):
                self.logger.debug(f"[DQN][{self.chosen_action}] Node {target_node} not DANGER.")
            if not any(p in self.products for p in o.products):
                self.logger.debug(f"[DQN][{self.chosen_action}] Node {target_node} no products '{o.products}' supported.")
                o.attackable = False
            if vuln.cve_id not in self.cve_list:
                self.logger.debug(f"[DQN][{self.chosen_action}] '{vuln.cve_id}' not in KEV/Metasploit list.")
                vuln.exploitable = False

            prob = self.calculate_success_probability(target_node, self.chosen_action, self.chosen_vuln)
            success = (np.random.rand() < prob)

            if (success):
                self.logger.debug(f"[DQN][{self.chosen_action}] Node {target_node} - execute {self.chosen_action}.")
                o.attackable = True
                vuln.exploitable = True
            else:
                self.logger.debug(f"[DQN][{self.chosen_action}] Node {target_node} - failed {self.chosen_action}.")
                if self.env.current_node is None:
                    self.failed_starting_nodes.add(target_node)
                    self.attack_time_remaining = -2
                else:
                    self.attack_time_remaining = -1
                    if target_node not in self.failed_visited_nodes:
                        self.failed_visited_nodes.append(target_node)

            return EXPLOIT, success

        # LATERAL MOVEMENT: move to a neighboring node using stolen credentials
        if self.chosen_action == LATERAL_MOVEMENT:

            prob = self.calculate_success_probability(target_node, self.chosen_action, self.chosen_vuln)
            success = (np.random.rand() < prob)

            if (success):
                self.logger.debug(f"[DQN][{self.chosen_action}] Node {target_node} - execute {self.chosen_action}.")
            else:
                self.logger.debug(f"[DQN][{self.chosen_action}] Node {target_node} - failed {self.chosen_action}.")
                self.attack_time_remaining = -1

            return LATERAL_MOVEMENT, success


        # --- Already-compromised node: post-exploitation actions ---
        if self.chosen_action in (CREDENTIALS_THEFT, PRIVILEGE_ESCALATION, PERSISTENCE, EXFILTRATION, WIPER):
            prob = self.calculate_success_probability(target_node, self.chosen_action, self.chosen_vuln)
            success = (np.random.rand() < prob)

            if (success):
                self.logger.debug(f"[DQN][_execute_action_for_node] Node {target_node} - execute {self.chosen_action}.")
            else:
                self.logger.debug(f"[DQN][_execute_action_for_node] Node {target_node} - failed {self.chosen_action}.")
                self.attack_time_remaining = -1

            return self.chosen_action, success


    # ------------------------------------------------------------------ #
    # DQN action selection (epsilon-greedy) over (node, action, vuln) triples
    # ------------------------------------------------------------------ #
    def choose_action_dqn(self, state, possible_pairs):
        """
        possible_pairs: list of (node, action, vuln) triples.
        Exploration: picks a random triple uniformly.
        Exploitation:
            - builds a Q-value mask over valid (node, action) flat indices
            - selects the best flat index
            - among all triples with that (node, action), picks the one with the highest CVSS score
        """
        # Exploration: random triple
        if self.training and random.random() <= self.epsilon:
            pair = random.choice(possible_pairs)
            self.logger.debug(f"[DQN][choose_action_dqn] EXPLORATION: random triple {pair}.")
            return pair

        # Exploitation: Q-value guided selection
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_all = self.model(state_tensor).cpu().numpy().squeeze()  # [num_nodes * 9]

        # Initialize mask to -inf; only valid (node, action) pairs receive their real Q-value
        mask = np.full_like(q_all, -1e9, dtype=np.float32)
        flat_to_pairs = {}

        for (n, a, v) in possible_pairs:
            if   a == SCAN:                 a_idx = 0
            elif a == EXPLOIT:              a_idx = 1
            elif a == LATERAL_MOVEMENT:     a_idx = 2
            elif a == CREDENTIALS_THEFT:    a_idx = 3
            elif a == PRIVILEGE_ESCALATION: a_idx = 4
            elif a == PERSISTENCE:          a_idx = 5
            elif a == EXFILTRATION:         a_idx = 6
            elif a == WIPER:                a_idx = 7
            elif a == DOS:                  a_idx = 8
            else:
                continue

            flat = n * NUM_NODE_ACTIONS + a_idx
            mask[flat] = q_all[flat]
            flat_to_pairs.setdefault(flat, []).append((n, a, v))

        best_flat = int(np.argmax(mask))
        best_list = flat_to_pairs.get(best_flat, None)

        filtered = [
            (n, a, v) for (n, a, v) in best_list
            if getattr(v, "cve_id", None) in self.cve_list
            and any(p in self.products for p in getattr(v, "products", []))
        ]

        # Fall back to unfiltered list if no candidate matches known CVEs/products
        candidates = filtered if filtered else best_list

        # Prefer the vulnerability with the highest CVSS score among candidates
        node, action, vuln = max(
            candidates,
            key=lambda item: getattr(item[2], "cvss_score", 0.0)
        )
        self.logger.debug(
            f"[DQN][choose_action_dqn] EXPLOITATION: selected triple "
            f"(node={node}, action={action}, vuln={getattr(vuln, 'cve_id', None)})."
        )
        return node, action, vuln

    def calculate_success_probability(self, node, action, vuln) -> float:
        node_obj = self.env.network.nodes[node]["node"]

        # For exploit and DoS: node must be in danger state with an unpatched, known CVE
        if action in (EXPLOIT, DOS):
            if node_obj.state != NODE_DANGER or vuln.patched:
                return 0.0

            if vuln.cve_id in self.cve_list:
                return self.success_rate

            if any(p in self.products for p in getattr(node_obj, "products", [])):
                return self.success_rate * 0.8

            return 0.0

        if action == LATERAL_MOVEMENT:
            return self.success_rate * 0.8

        # Post-exploitation actions require prior access to the node
        if not node_obj.attacker_has_access:
            return 0.0

        # Fixed success probabilities per post-exploitation action
        action_probs = {
            PERSISTENCE: 0.7 + (0.2 if node_obj.attacker_has_privileges else 0.0),
            CREDENTIALS_THEFT: 0.7,
            PRIVILEGE_ESCALATION: 0.7,
            EXFILTRATION: 0.7,
            WIPER: 0.7,
        }

        return action_probs.get(action, 0.0)


    def update_attack_time(self, vuln):
        if vuln is None:
            self.attack_time_remaining = 1
            return

        attack_complexity = getattr(vuln, "attack_complexity").upper()

        if attack_complexity == "HIGH":
            self.attack_time_remaining = 2   # complex attack takes two time steps
        elif attack_complexity == "LOW":
            self.attack_time_remaining = 1   # simple attack takes one time step

    # ------------------------------------------------------------------ #
    # Replay memory & training
    # ------------------------------------------------------------------ #
    def add_to_memory(self, state, node_action, post_action, reward, next_state, done):
        """
        node_action: index of the chosen node (self.chosen_node)
        post_action: action actually executed (SCAN/EXPLOIT/POST/DOS)
        """
        if not self.training or node_action is None:
            return

        n = int(node_action)
        if   post_action == SCAN:                 a_idx = 0
        elif post_action == EXPLOIT:              a_idx = 1
        elif post_action == LATERAL_MOVEMENT:     a_idx = 2
        elif post_action == CREDENTIALS_THEFT:    a_idx = 3
        elif post_action == PRIVILEGE_ESCALATION: a_idx = 4
        elif post_action == PERSISTENCE:          a_idx = 5
        elif post_action == EXFILTRATION:         a_idx = 6
        elif post_action == WIPER:                a_idx = 7
        elif post_action == DOS:                  a_idx = 8
        else:
            return

        flat = n * NUM_NODE_ACTIONS + a_idx
        self.memory.append((state, flat, float(reward), next_state, float(done)))
        self.train_counter += 1
        if self.train_counter % self.train_interval == 0:
            self.train()

    def train(self):
        if not self.training or len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, flats, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(states).to(self.device)
        flats = torch.LongTensor(flats).to(self.device)
        rewards = torch.FloatFloatTensor(rewards).to(self.device) if False else torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_all = self.model(states)
        q_sa = q_all.gather(1, flats.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            main_next = self.model(next_states)
            target_next = self.target_model(next_states)
            best_next = main_next.argmax(dim=1)
            max_q_next = target_next.gather(1, best_next.unsqueeze(1)).squeeze(1)
            target = rewards + (1.0 - dones) * self.gamma * max_q_next

        loss = nn.SmoothL1Loss()(q_sa, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5)
        self.optimizer.step()

        self.loss_history.append(loss.item())
        self.step_counter += 1
        if self.step_counter % self.target_update_freq == 0:
            self.update_target_network()
            self.logger.debug("[DQN][TRAIN] Target network updated.")

    def decay_epsilon(self):
        if self.training:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
