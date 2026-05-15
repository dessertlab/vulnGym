import gymnasium as gym
import numpy as np
import networkx as nx
import random
import copy
import NetworkNode
import simulation_parameters as sp
import os

from abc import ABC
from collections import deque
from macro import *
from rewards import *

class NetworkExplorationEnv(gym.Env, ABC):

    def __init__(self, num_nodes, dataset_vulns, seed, logger, goal, topology, phishing=False):
        super(NetworkExplorationEnv, self).__init__()

        self.logger = logger
        self.num_nodes = int(num_nodes)
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

        # dataset settings
        self.dataset_vulns = dataset_vulns
        self.goal = goal

        self.simulation_time_limit = sp.TIME_LIMIT
        self.current_time = 0

        self.node_delays = {node: 0 for node in range(self.num_nodes)}

        self.node_objects = {}

        self.network = nx.Graph()
        self.topology = topology

        self.zone_ranges = {
            "external": (1, 3),
            "dmz": (3, 6),
            "internal": (6, 9),
            "database": (9, 10)
        }

        self.build_network()
        self.failed_attack_nodes = []

        # Betweenness centrality computed once after topology is built; used as a static node feature
        self.centrality = nx.betweenness_centrality(self.network)

        self.hidden_vulnerabilities = {}
        self.node_vulnerability_times = {}

        self.phishing = phishing

        self.state_size_attacker = len(self.get_state_for_attacker())
        self.action_size = self.num_nodes

        self.compromised_nodes = set()

        self.time_to_goal_ATTACKER = None
        self.time_to_patch = []

        self.last_episodes_vulnerabilities_exploited = deque(maxlen=sp.NUM_EPISODES_TO_TRACK)

    # -------------------- Reset environment --------------------
    def reset(self, seed):
        """Reset the environment to its initial configuration for a new simulation episode."""
        self.logger.debug("[NETWORK_E][RESET] Environment reset. New simulation.")

        np.random.seed(self.seed)
        random.seed(self.seed)

        self.current_node = None
        self.current_time = 0
        self.node_delays = {node: 0 for node in range(self.num_nodes)}

        self.hidden_vulnerabilities, self.node_vulnerability_times = self.schedule_vulnerabilities(self.dataset_vulns, seed)

        for _, node in self.node_objects.items():
            node.reset()

        self.failed_attack_nodes = []
        self.state_size = len(self.get_state())
        self.action_size = self.num_nodes

        self.compromised_nodes.clear()

        self.time_to_goal_ATTACKER = None
        self.time_to_patch = []

        self.logger.debug("[NETWORK_E][RESET] Initial state successfully regenerated.")


    # -------------------- Multi-vulnerability helpers --------------------

    @staticmethod
    def _first_unpatched_vuln(node_obj):
        """Return the first unpatched vulnerability on the node, or None if all are patched."""
        for v in getattr(node_obj, "vulnerabilities", []):
            if not getattr(v, "patched", False):
                return v
        return None


    @staticmethod
    def _first_exploited_vuln(node_obj):
        """Return the first exploited and unpatched vulnerability on the node, or None."""
        for v in getattr(node_obj, "vulnerabilities", []):
            if getattr(v, "exploited", False) and not getattr(v, "patched", False):
                return v
        return None



    # -------------------- Build network --------------------

    def build_network(self):
        if self.topology == "tree":
            self.topology_tree()
        elif self.topology == "layered":
            self.topology_layered()
        else:
            raise ValueError(f"Unknown topology: {self.topology}")

        # with open(os.path.join("./info_network", "node_products.txt"), "w") as f:
        #     for node_id, node in self.node_objects.items():
        #         f.write(f"Node ID: {node_id}, Zone: {node.zone}, Assigned Products: {', '.join(node.products)}, Phishing Possible: {'Yes' if node.phishing else 'No'}\n")


    def topology_tree(self):
        seed = self.seed
        np.random.seed(seed)

        self.network.clear()
        self.node_objects = {}

        # -------------------------
        # 1. LAYER CONFIGURATION
        # -------------------------
        level_config = {
            "external": {
                "n_nodes": int(self.num_nodes * 0.1),
                "n_subnets": 1
            },
            "dmz": {
                "n_nodes": int(self.num_nodes * 0.3),
                "n_subnets": 3
            },
            "internal": {
                "n_nodes": int(self.num_nodes * 0.4),
                "n_subnets": 4
            },
            "database": {
                "n_nodes": int(self.num_nodes * 0.2),
                "n_subnets": 3
            }
        }

        node_id = 0
        levels = {}

        # -------------------------
        # 2. CREATE LAYERS AND SUBNETS
        # -------------------------
        def create_level(zone, n_nodes, n_subnets):
            nonlocal node_id

            nodes = list(range(node_id, node_id + n_nodes))
            node_id += n_nodes

            subnets = np.array_split(nodes, n_subnets)
            subnet_list = []

            for subnet_id, subnet in enumerate(subnets):
                subnet = list(subnet)

                for i in subnet:
                    node = NetworkNode.create_node(
                        i,
                        zone,
                        np.random.uniform(*self.zone_ranges[zone])
                    )

                    node.zone = zone
                    node.subnet_id = subnet_id
                    node.is_shared = False

                    self.node_objects[i] = node
                    self.network.add_node(i, node=node)

                # Intra-subnet edges only; cross-subnet links are added in the next step
                for i in range(len(subnet)):
                    for j in range(i + 1, len(subnet)):
                        self.network.add_edge(subnet[i], subnet[j])

                subnet_list.append(subnet)

            return subnet_list

        for zone, cfg in level_config.items():
            levels[zone] = create_level(
                zone,
                cfg["n_nodes"],
                cfg["n_subnets"]
            )

        # -------------------------
        # 3. CROSS-LAYER CONNECTIONS
        # -------------------------
        def connect_subnets_sparse(lower_subnets, upper_subnets, p_active=0.4):
            """
            Connect lower-layer subnets to upper-layer subnets sparsely.
            Only a fraction (p_active) of nodes in each lower subnet act as gateways.
            """

            for lower in lower_subnets:

                for i in lower:

                    # Only a fraction of lower-layer nodes become gateways to the next layer
                    if np.random.rand() > p_active:
                        continue

                    # Pick a random upper subnet to connect to
                    upper = upper_subnets[np.random.randint(len(upper_subnets))]

                    # Connect this gateway to every node in the chosen upper subnet
                    for j in upper:
                        self.network.add_edge(i, j)
                        self.network[i][j]["layer_edge"] = True
                        self.node_objects[j].is_shared = True

        connect_subnets_sparse(levels["external"], levels["dmz"], p_active=0.6)
        connect_subnets_sparse(levels["dmz"], levels["internal"], p_active=0.4)
        connect_subnets_sparse(levels["internal"], levels["database"], p_active=0.3)

        # -------------------------
        # 4. ATTACKER ENTRY POINTS
        # -------------------------
        external_nodes = [n for subnet in levels["external"] for n in subnet]

        self.starting_nodes = external_nodes.copy()
        self.current_node = None
        self.attacker_position = None
        self.forced_move = False

        # -------------------------
        # 5. VULNERABILITY SCHEDULING
        # -------------------------
        self.hidden_vulnerabilities, self.node_vulnerability_times = \
            self.schedule_vulnerabilities(self.dataset_vulns, self.seed)

    def topology_layered(self):

        seed = self.seed
        np.random.seed(seed)

        self.network.clear()
        self.node_objects = {}

        # -------------------------
        # 1. ZONE SPLIT
        # -------------------------
        a = max(1, int(self.num_nodes * 0.1))
        b = max(a, int(self.num_nodes * 0.4))
        c = max(b, int(self.num_nodes * 0.9))

        external_nodes = list(range(0, a))
        dmz_nodes = list(range(a, b))
        internal_nodes = list(range(b, c))
        database_nodes = list(range(c, self.num_nodes))

        # -------------------------
        # 2. CREATE NODES
        # -------------------------
        def create_nodes(node_list, zone):
            for i in node_list:
                self.node_objects[i] = NetworkNode.create_node(
                    i, zone, np.random.uniform(*self.zone_ranges[zone])
                )
                self.network.add_node(i, node=self.node_objects[i])

        create_nodes(external_nodes, "external")
        create_nodes(dmz_nodes, "dmz")
        create_nodes(internal_nodes, "internal")
        create_nodes(database_nodes, "database")

        # -------------------------
        # 3. FULLY CONNECT LAYERS
        # -------------------------
        def fully_connect(nodes):
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    self.network.add_edge(nodes[i], nodes[j])

        fully_connect(external_nodes)
        fully_connect(dmz_nodes)
        fully_connect(internal_nodes)
        fully_connect(database_nodes)

        # -------------------------
        # 4. CROSS-LAYER CONNECTIONS
        # -------------------------
        def connect_layers_fixed_shared(lower_nodes, upper_nodes, k, label=None):
            """
            Select k gateway nodes in the upper layer and connect all lower nodes to them.
            Using a fixed shared set (chosen once) ensures a consistent bottleneck topology.
            """

            if len(lower_nodes) == 0 or len(upper_nodes) == 0:
                return []

            k = min(k, len(upper_nodes))

            # Gateway nodes are chosen once per layer pair to keep the topology deterministic
            shared_nodes = list(np.random.choice(upper_nodes, k, replace=False))

            for i in lower_nodes:
                for j in shared_nodes:
                    self.network.add_edge(i, j)

            for j in shared_nodes:
                self.node_objects[j].is_shared = True
                if label:
                    self.node_objects[j].shared_type = label

            return shared_nodes

        # -------------------------
        # 5. APPLY CONNECTIONS
        # -------------------------

        # External → DMZ
        self.dmz_shared_nodes = connect_layers_fixed_shared(
            external_nodes,
            dmz_nodes,
            k=3,
            label="dmz"
        )

        # DMZ → INTERNAL
        self.internal_shared_nodes = connect_layers_fixed_shared(
            dmz_nodes,
            internal_nodes,
            k=4,
            label="internal"
        )

        # INTERNAL → DATABASE
        self.database_shared_nodes = connect_layers_fixed_shared(
            internal_nodes,
            database_nodes,
            k=3,
            label="database"
        )

        # -------------------------
        # 6. INIT STATE
        # -------------------------
        self.starting_nodes = external_nodes.copy()
        self.current_node = None
        self.attacker_position = None
        self.forced_move = False

        self.hidden_vulnerabilities, self.node_vulnerability_times = self.schedule_vulnerabilities(self.dataset_vulns, self.seed)


    def schedule_vulnerabilities(self, vulnerabilities_list, seed):

        hidden = {n: [] for n in range(self.num_nodes)}
        times = {}

        node_vuln_associations = {n: [] for n in range(self.num_nodes)}

        vulnerabilities_copy = vulnerabilities_list.copy()

        # Shuffle to randomize which vulnerabilities each node receives across episodes
        rng = random.Random(seed)
        rng.shuffle(vulnerabilities_copy)

        # Match vulnerabilities to nodes by product compatibility
        for vuln in vulnerabilities_copy:
            for node_id, node in self.node_objects.items():
                if any(p in (vuln.products) for p in (node.products)):
                    node_vuln_associations[node_id].append(vuln)

        for node_id, vulns in node_vuln_associations.items():
            for vuln in vulns[:sp.MAX_VULNS_PER_NODE]:
                hidden.setdefault(node_id, []).append(vuln)
                times.setdefault(node_id, []).append((vuln, vuln.release_time))

        return hidden, times


    def get_nodes_by_state(self, state):
        return [node_id for node_id, obj in self.node_objects.items() if obj.state == state]


    # -------------------- State representation --------------------
    def get_state_for_attacker(self):
        N = self.num_nodes
        nodes = self.node_objects

        state = []

        for n in range(N):
            obj = nodes[n]

            # Base flags
            is_danger = 1 if obj.state == NODE_DANGER else 0

            is_compromised = (
                1 if getattr(obj, "attacker_has_access", False)
                else 0
            )

            is_neighbor_from_stolen = (
                1 if any(
                    getattr(neighbor, "credentials_stolen", False)
                    for neighbor in self.network.neighbors(n)
                ) else 0
            )

            # Static features (normalized)
            importance = obj.importance / 10
            centrality = self.centrality.get(n, 0.0)

            # Dynamic vulnerability features
            vulns = obj.vulnerabilities
            if len(vulns) > 0:
                cvss_values = [v.cvss_score for v in vulns]
                cvss_max = max(cvss_values) / 10
            else:
                cvss_max = 0.0

            # Build node feature vector
            node_features = [
                is_danger,
                is_compromised,
                is_neighbor_from_stolen,
                importance,
                centrality,
                cvss_max,
            ]

            state.extend(node_features)

        return tuple(state)

    def get_state(self):

        state_attacker = self.get_state_for_attacker()
        state_defender = 0

        return [state_attacker,state_defender]


    # -------------------- Remote scan --------------------
    def perform_remote_scan(self, node, supported_products=None, supported_vulns=None):
        """
        Execute a remote scan on the target node.
        Always returns True (scans are assumed to succeed).
        Sets obj.scanned = True and determines attackability from product match only.
        Each unpatched vulnerability is then marked exploitable iff its CVE ID is in supported_vulns.
        """
        obj = self.node_objects[node]

        # Always mark as scanned regardless of product match
        obj.scanned = True

        if any(p in (supported_products) for p in obj.products):
            obj.attackable = True
            self.logger.debug(f"[NETWORK_E][REMOTE_SCAN] Node {node} marked as attackable.")
        else:
            obj.attackable = False
            self.logger.debug(f"[NETWORK_E][REMOTE_SCAN] Node {node} marked as NOT attackable.")

        # Only consider unpatched vulnerabilities for exploitability assessment
        unpatched_vulns = [
            v for v in getattr(obj, "vulnerabilities", [])
            if not getattr(v, "patched", False)
        ]

        # Normalize CVE IDs for case-insensitive comparison
        my_vulns = [str(v).lower().strip() for v in (supported_vulns)]

        # Mark each unpatched vuln as exploitable or not based on CVE match
        if unpatched_vulns and my_vulns:
            for vuln in unpatched_vulns:
                cve_id = str(getattr(vuln, "cve_id", "") or "").lower().strip()

                if cve_id in my_vulns:
                    vuln.exploitable = True
                    self.logger.debug(f"[NETWORK_E][REMOTE_SCAN] Node {node} vuln {cve_id} marked as exploitable.")
                else:
                    vuln.exploitable = False
                    self.logger.debug(f"[NETWORK_E][REMOTE_SCAN] Node {node} vuln {cve_id} marked as NOT exploitable.")

        return True


    # -------------------- Post-exploit action helpers --------------------
    def _valid_post_actions_for_node(self, node, vulns):
        """
        Return valid post-exploitation (node, action, vuln) triples for an already-compromised node.
        Post-actions require at least one exploited vulnerability; each one-shot action is only
        offered if it has not already been performed on the node.
        """
        obj = self.node_objects[node]
        valid = []

        if not vulns:
            return valid

        for vuln in vulns:

            # Only offer each one-shot post-exploitation action if not already performed
            if REMOTE_CONTROL in vuln.vulnerability_type :
                if not getattr(obj, "credentials_stolen", False):
                    valid.append((node, CREDENTIALS_THEFT, vuln))
                if not getattr(obj, "attacker_has_privileges", False):
                    valid.append((node, PRIVILEGE_ESCALATION, vuln))
                if not getattr(obj, "attacker_persistent", False):
                    valid.append((node, PERSISTENCE, vuln))
                if not getattr(obj, "data_exfiltrated", False):
                    valid.append((node, EXFILTRATION, vuln))
                if not getattr(obj, "is_wiped", False):
                    valid.append((node, WIPER, vuln))
            # DoS action is available only while the node is still online
            elif DOS in vuln.vulnerability_type and not obj.is_off:
                valid.append((node, DOS, vuln))

        return valid



    # -------------------- Step (exploit + post-action) --------------------
    def step(self, attacker_action, defender_action):
        """
        Advance the simulation by one time step: apply attacker and defender actions,
        reveal newly disclosed vulnerabilities, compute rewards, and check terminal conditions.
        """
        attacker_reward = 0
        defender_reward = 0

        attacker_done = False
        defender_done = False

        self.current_time += 1

        # End the episode when the time limit is reached
        if self.current_time > self.simulation_time_limit:

            state = self.get_state()

            #attacker_reward -= reward_TIME_LIMIT_EXCEEDED
            #defender_reward += reward_TIME_LIMIT_EXCEEDED

            defender_done = True

            self.logger.debug("[NETWORK_E][STEP_SIMULATION] Simulation ended.")

            return state, [attacker_reward, defender_reward], [attacker_done, defender_done]

        # TODO: handle the case where both agents act on the same node simultaneously
        if attacker_action is not None and defender_action:
            attacker_node = attacker_action[0]
            action = attacker_action[1]
            attacker_vuln = attacker_action[2]

            if action != SCAN and action != LATERAL_MOVEMENT:
                for node, vuln, success in defender_action:
                    if attacker_node == node and attacker_vuln.cve_id == vuln.cve_id:
                        self.logger.debug(
                            f"[NETWORK_E][STEP_SIMULATION] Action conflict on node {attacker_node}"
                        )

                        if success:
                            attacker_action[3] = False
                        break

        self.reveal_vulnerabilities()

        if self.forced_move:
            self.forced_move = False
            self.logger.debug("[NETWORK_E][STEP_SIMULATION] Attacker forced to move.")

            #attacker_action[1] = NO_ACTION
            #action = NO_ACTION
            attacker_reward -= REWARD_FORCED_MOVE

        if attacker_action is not None:
            attacker_reward += self.compute_attacker_reward(attacker_action)


        if defender_action is not None:
            for action in defender_action:
                defender_reward += self.compute_defender_reward(action)

        state = self.get_state()

        if self.time_to_goal_ATTACKER is None:

            attacker_done = self.check_attacker_goal()

            if attacker_done:
                attacker_reward += REWARD_DONE
                defender_reward -= REWARD_DONE
                self.time_to_goal_ATTACKER = self.current_time
                self.logger.debug(f"[NETWORK_E][STEP_SIMULATION] Done. Final reward: {attacker_reward}.")

        return state, [attacker_reward, defender_reward], [attacker_done, defender_done]

    def check_attacker_goal(self):

        if self.goal == "data_exfiltration":
            return self.is_top_database_compromised()

        elif self.goal == "db_denial_of_service":
            return self.is_top_database_dos()

        # elif self.goal == "credential_collection":
        #     return self.credentials_stolen >= 3

        # elif self.goal == "lateral_movement":
        #     compromised = sum(1 for o in self.node_objects.values() if o.attacker_has_access)
        #     return compromised >= int(self.num_nodes * 0.5)

        # elif self.goal == "crypto_mining":
        #     return self.persistence_nodes >= 3

        # elif self.goal == "sabotage":
        #     return self.critical_nodes_destroyed >= 1

        return False



    def compute_attacker_reward(self, attacker_action):

        node, action, vulnerability, success = attacker_action

        reward = 0.0

        obj = None
        if node is not None:
            obj = self.network.nodes[node]["node"]

        # Failure: no reward; still mark node as visited
        if not success:
            self.logger.debug(f"[REWARD] {action} FAILED → reward = 0")

            if obj is not None:
                obj.visited = True

            self.attacker_position = self.current_node
            return reward

        # Success: scale reward by action importance and node importance
        action_importance = ACTION_IMPORTANCE[self.goal].get(action.upper())

        reward = action_importance * obj.importance

        self.logger.debug(
            f"[REWARD] {action} SUCCESS → action_imp={action_importance}, "
            f"node_imp={obj.importance}, reward={reward:.2f}"
        )

        # Apply side effects of the successful action to the environment state

        if action == SCAN and obj is not None:
            obj.scanned = True

        elif action == EXPLOIT and obj is not None:
            obj.attacker_has_access = True
            self.compromised_nodes.add(node)

            if vulnerability is not None:
                vulnerability.exploited = True

            self.current_node = node

        elif action == LATERAL_MOVEMENT and obj is not None:
            obj.attacker_has_access = True
            self.compromised_nodes.add(node)
            self.current_node = node

        elif action == CREDENTIALS_THEFT:
            obj.credentials_stolen = True
            self.current_node = node

        elif action == PRIVILEGE_ESCALATION and obj is not None:
            obj.attacker_has_privileges = True
            self.current_node = node

        elif action == PERSISTENCE:
            obj.attacker_persistent = True
            self.current_node = node

        elif action == EXFILTRATION:
            obj.data_exfiltrated = True
            self.current_node = node

        elif action == WIPER:
            obj.is_wiped = True
            self.current_node = node

        elif action == DOS:
            if obj.zone == "database" and self.goal == "db_denial_of_service":
                reward += 2.0
            obj.is_off = True
            if vulnerability is not None:
                vulnerability.exploited = True

        return reward


    def reveal_vulnerabilities(self):
        """
        Reveal hidden vulnerabilities at their scheduled release time.
        Vulnerability types must already be set when the dataset is loaded.
        Supports multi-vulnerability nodes: each newly revealed vuln is appended
        to the node's list and may transition the node state to DANGER.
        """
        for node, vulns_tuple in list(self.node_vulnerability_times.items()):
            obj = self.network.nodes[node]["node"]

            if obj.is_off:
                continue

            for vulnerability, release_time in vulns_tuple:
                if release_time == self.current_time and vulnerability not in obj.vulnerabilities:
                    obj.vulnerabilities.append(copy.copy(vulnerability))

                    # A newly revealed unpatched vuln must transition the node to DANGER
                    NetworkNode.recompute_state(obj)

                    self.logger.debug(
                        f"[NETWORK_E][REVEAL] Node {node} CVE {vulnerability.cve_id}, "
                        f"CVSS {vulnerability.cvss_score}, expo {vulnerability.exploitability}, "
                        f"type {vulnerability.vulnerability_type}"
                    )

                    # Remove the now-revealed vuln from the hidden pool
                    if node in self.hidden_vulnerabilities and vulnerability in self.hidden_vulnerabilities[node]:
                        self.hidden_vulnerabilities[node].remove(vulnerability)
                        if not self.hidden_vulnerabilities[node]:
                            del self.hidden_vulnerabilities[node]


    # -------------------- Patch node --------------------
    def compute_defender_reward(self, defender_action):
        """
        Apply a patch to one vulnerability on the target node and compute the defender reward.
        If a specific vulnerability is passed it is patched; otherwise the first unpatched one is used.
        Node state rules after patching:
        - Remaining unpatched vulns → DANGER
        - No unpatched vulns remaining → PATCHED
        - attacker_persistent=True → node stays DANGER and attacker retains access
        """

        node, vulnerability, success = defender_action

        obj = self.network.nodes[node]["node"]

        if not success:

            if obj.is_off:
                self.logger.debug(f"[NETWORK_E][RESOLVE_NODE] Node {node} cannot be patched (DoS/off).")
                return REWARD_NO_PATCHABLE_NODE

            self.logger.debug(f"[NETWORK_E][RESOLVE_NODE] Patch attempt failed on node {node}.")
            return REWARD_PATCH_FAILURE

        if success :

            vulnerability.patched = True
            reward = 0

            # Clear any queued delay on this node
            self.node_delays[node] = 0

            previous_access = obj.attacker_has_access

            if any(vulnerability.cve_id in ep for ep in self.last_episodes_vulnerabilities_exploited):
                reward += REWARD_PATCH_ON_EXPLOITED_VULN

            # Determine whether the attacker retains access after the patch
            if obj.attacker_persistent:
                # Persistence keeps attacker access even after patching
                obj.attacker_has_access = True
                attacker_access_after_patch = True
                self.logger.debug(
                    f"[NETWORK_E][RESOLVE_NODE] Node {node} persistent: patch does not remove access."
                )
            else:
                if vulnerability.exploited:
                    obj.attacker_has_access = False
                    obj.attacker_has_privileges = False
                    # obj.attacker_persistent = False
                    # obj.data_exfiltrated = False
                    # obj.credentials_stolen = False
                attacker_access_after_patch = False

            # Recompute node state based on remaining unpatched vulnerabilities
            NetworkNode.recompute_state(obj)

            if obj.state == NODE_SAFE and getattr(node, "node_id", None) in self.compromised_nodes:
                self.compromised_nodes.remove(getattr(node, "node_id", None))

            # Force the attacker to move if they are currently on this node and just lost access
            if self.attacker_position is not None and self.attacker_position == node and self.attacker_position == self.current_node and not attacker_access_after_patch:
                self.force_attacker_to_move()
                reward += REWARD_FORCED_MOVE
                self.logger.debug(
                    f"[NETWORK_E][RESOLVE_NODE] Attacker forced to leave node {node}."
                )

            # #score = ((vulnerability.cvss_score / 10.0) if vulnerability else 0.0) \
            # #        + (obj.importance / 10.0) + self.centrality.get(node, 0.0)
            # #reward += reward_MULTIPLICATOR_FACTOR_DEFENDER * score

            self.logger.debug(
                f"[NETWORK_E][RESOLVE_NODE] Node {node} patched. "
                f"Previous access: {previous_access}, access_after_patch: {attacker_access_after_patch}, "
                f"state={obj.state}. Reward={reward:.2f}"
            )
            return reward


    def force_attacker_to_move(self):
        """
        Relocate the attacker when their current node is patched or taken offline.
        Picks a random node where the attacker still has active access; sets current_node
        to None if no such node exists (attacker loses their foothold entirely).
        """
        self.forced_move = True

        compromised_nodes = [
            n for n, obj in self.node_objects.items()
            if obj.attacker_has_access and not obj.is_off
        ]
        if compromised_nodes:
            self.current_node = random.choice(compromised_nodes)
            self.logger.debug(f"[NETWORK_E][FORCE_MOVE] Attacker moved to node {self.current_node}.")
        else:
            self.current_node = None
            self.logger.debug("[NETWORK_E][FORCE_MOVE] No remaining access: attacker loses position.")




    # -------------------- Node statistics --------------------
    def get_exploited_nodes(self):
            """List of node IDs with any active attacker foothold (access, DoS, wipe, exfil, etc.)."""
            return [nid for nid, obj in self.node_objects.items() if obj.attacker_has_access or obj.is_off or obj.is_wiped or obj.data_exfiltrated or obj.credentials_stolen or obj.attacker_persistent or obj.attacker_has_privileges ]

    def get_patched_nodes(self):
            """List of node IDs where all exploited vulnerabilities have been patched."""
            patched_nodes = []
            for nid, obj in self.node_objects.items():

                vulns = obj.vulnerabilities

                # Condition A: all exploited vulns must be patched
                all_exploited_are_patched = all(
                    (not v.exploited) or v.patched
                    for v in vulns
                )

                # Condition B: at least one vulnerability is patched
                any_patched = any(v.patched for v in vulns)

                add = all_exploited_are_patched and any_patched

                if add and not obj.attacker_persistent:
                    patched_nodes.append(nid)

            return patched_nodes

    def count_exploited_nodes(self):
            """Number of nodes with an active attacker foothold."""
            return len(self.get_exploited_nodes())

    def count_patched_nodes(self):
            """Number of nodes with at least one patched vulnerability."""
            return len(self.get_patched_nodes())


    # -------------------- Vulnerability statistics --------------------

    def get_exploited_vulnerabilities(self):
            """Return a list of (node_id, vuln) tuples for all exploited vulnerabilities."""
            exploited = []
            for nid, obj in self.node_objects.items():
                for v in obj.vulnerabilities:
                    if getattr(v, "exploited", False):
                        exploited.append((nid, v))
            return exploited

    def get_patched_vulnerabilities(self):
            """Return a list of (node_id, vuln) tuples for all patched vulnerabilities."""
            patched = []
            for nid, obj in self.node_objects.items():
                for v in obj.vulnerabilities:
                    if getattr(v, "patched", False):
                        patched.append((nid, v))
            return patched

    def count_exploited_vulnerabilities(self):
            """Total number of exploited vulnerabilities across all nodes."""
            return len(self.get_exploited_vulnerabilities())

    def count_patched_vulnerabilities(self):
            """Total number of patched vulnerabilities across all nodes."""
            return len(self.get_patched_vulnerabilities())

    # -------------------- Zone statistics --------------------

    def compute_zone_node_stats(env):
        """
        Count total, exploited, and patched nodes per network zone.
        Returns: {"external": {"total": n, "exploited": x, "patched": y}, ...}
        """

        ZONES = ["external", "dmz", "internal", "database"]

        exploited_nodes = set(env.get_exploited_nodes())
        patched_nodes   = set(env.get_patched_nodes())

        stats = {z: {"total": 0, "exploited": 0, "patched": 0} for z in ZONES}

        for nid, obj in env.node_objects.items():
            zone = getattr(obj, "zone", "unknown")

            # Handle zones outside the standard set
            if zone not in stats:
                stats[zone] = {"total": 0, "exploited": 0, "patched": 0}

            stats[zone]["total"] += 1

            if nid in exploited_nodes:
                stats[zone]["exploited"] += 1

            if nid in patched_nodes:
                stats[zone]["patched"] += 1

        return stats

    def compute_zone_vuln_stats(env):
        """
        Count total, exploited, and patched vulnerabilities per network zone.
        Returns: {"external": {"total": n, "exploited": x, "patched": y}, ...}
        """

        ZONES = ["external", "dmz", "internal", "database"]

        stats = {z: {"total": 0, "exploited": 0, "patched": 0} for z in ZONES}

        for nid, obj in env.node_objects.items():
            zone = getattr(obj, "zone", "unknown")

            if zone not in stats:
                stats[zone] = {"total": 0, "exploited": 0, "patched": 0}

            for v in obj.vulnerabilities:
                stats[zone]["total"] += 1

                if getattr(v, "exploited", False):
                    stats[zone]["exploited"] += 1

                if getattr(v, "patched", False):
                    stats[zone]["patched"] += 1

        return stats

    def is_top_database_compromised(self):
        database_nodes = [
            node_id for node_id, node_obj in self.node_objects.items()
            if node_obj.zone == "database"
        ]
        if not database_nodes:
            return False

        # Goal met when enough database nodes have been exfiltrated or wiped
        exfiltrated_count = sum(
            1 for node_id in database_nodes
            if self.node_objects[node_id].data_exfiltrated or self.node_objects[node_id].is_wiped
        )

        return exfiltrated_count >= sp.TARGET_GOALS


    def is_top_database_dos(self):
        database_nodes = [
            node_id for node_id, node_obj in self.node_objects.items()
            if node_obj.zone == "database"
        ]

        if not database_nodes:
            return False

        # Goal met when enough database nodes have been taken offline
        dos_count = sum(
            1 for node_id in database_nodes
            if self.node_objects[node_id].is_off
        )

        return dos_count >= sp.TARGET_GOALS

    def calculate_NVI(self):

        num_exploited_nodes = sum(
            1 for obj in self.node_objects.values()
            if getattr(obj, "attacker_has_access", False)
        )
        nvi = (num_exploited_nodes / self.num_nodes) * 100

        return nvi
