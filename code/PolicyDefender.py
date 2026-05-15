import random
import math
import simulation_parameters as sp

from macro import *
from rewards import *
from collections import defaultdict


class PolicyDefender:

    def __init__(self, env, policy, cycle_duration=30, discovery_time=1, fix_time=7, logger=None, seed=None, success_probability=sp.SUCCESS_PROBABILITY_DEFENDER):

        random.seed(seed)

        self.logger = logger

        self.env = env
        self.policy = policy
        self.cycle_duration = cycle_duration
        self.discovery_time = discovery_time
        self.fix_time = fix_time

        self.current_day = 0

        self.success_probability = success_probability

        self.vulnerabilities_discovered = defaultdict(set)
        self.vulnerability_timestamps = {}
        self.vulnerabilities_exploited = set()
        self.vulns_in_backlog = []

        self.cve_patched = []

        self.chosen_node = None
        self.chosen_vuln = None

        self.patching_time_remaining = -1


    def reset(self, seed):

        random.seed(seed)   # reseed to get different behavior per episode

        self.logger.debug("[P_DEFENDER] Reset of the Defender.")

        self.current_day = 0
        self.vulnerabilities_discovered = defaultdict(set)
        self.vulnerability_timestamps = {}
        self.cve_patched = []
        self.vulns_in_backlog = []

        self.chosen_node = []
        self.chosen_vuln = None
        self.patching_time_remaining = -1

        self.env.last_episodes_vulnerabilities_exploited.append(self.vulnerabilities_exploited.copy())
        self.vulnerabilities_exploited.clear()


    def step(self):
        self.current_day += 1

        # Duration of one mini-cycle (discovery days + fix days)
        subcycle_length = self.discovery_time + self.fix_time

        # 1-based day index within the current sub-cycle
        day_in_subcycle = (self.current_day - 1) % subcycle_length + 1

        # Discovery phase: scan for new vulnerabilities
        if day_in_subcycle <= self.discovery_time:
            self.discovery_phase()

            if self.patching_time_remaining >= 0:
                action= self.resolution_phase()  # Continue patching if a patch is already in progress
            else:
                action = None

        else:
            if self.patching_time_remaining == -1:
               # Update chosen vuln & chosen nodes
               self.select_target_vulnerability()
            action = self.resolution_phase()

        # Reset counters at the end of the main cycle
        if self.current_day >= self.cycle_duration:
            self.logger.debug(f"[P_DEFENDER][CYCLE] End of the main cycle.")
            self.current_day = 0
            self.vulnerabilities_discovered.clear()
            self.vulnerability_timestamps = {}

        return action


    def discovery_phase(self):
        """
        Scan the network: register visible vulnerabilities that are not yet patched.
        Exclude off/unpatchable nodes; include ONLY unpatched vulnerabilities.
        """

        # Log the number of vulnerabilities in the backlog at the beginning of the discovery phase
        self.vulns_in_backlog.append(len(self.vulnerabilities_discovered.keys()))

        for node in self.env.network.nodes:
            obj = self.env.network.nodes[node]["node"]

            if getattr(obj, "is_off", False):
                # If the node is off, it cannot be scanned, so we skip it
                continue

            if obj.vulnerabilities:
                for vuln in obj.vulnerabilities:
                    if getattr(vuln, "patched", False):
                        # The vulnerability is already patched
                        continue

                    if getattr(vuln, "exploited", False):
                        # The vulnerability is already exploited
                        self.vulnerabilities_exploited.add(vuln.cve_id)


                    key = (node, vuln.cve_id)
                    if key not in self.vulnerability_timestamps:
                        self.vulnerability_timestamps[key] = {
                            "discovery": self.env.current_time,
                            "patch": None
                        }
                    self.vulnerabilities_discovered[vuln.cve_id].add(node)

        self.logger.debug(f"[P_DEFENDER][DISCOVER] The defender discovered {len(self.vulnerabilities_discovered)} unique vulnerabilities.")


    def resolution_phase(self):
        """
        Apply the patch to the selected vulnerability and nodes, if the patching time has elapsed.
        """
        if self.chosen_vuln is None:
            self.logger.debug("[P_DEFENDER][RESOLUTION] No vulnerability selected.")
            return None

        if self.patching_time_remaining > 0:
            self.logger.debug(f"[P_DEFENDER][RESOLUTION] Patching in progress for vulnerability {self.chosen_vuln} on nodes {self.chosen_node}. Time remaining: {self.patching_time_remaining} days.")
            self.patching_time_remaining -= 1
            return None  # Patch is still in progress

        if self.patching_time_remaining == 0:
            self.logger.debug(f"[P_DEFENDER][RESOLUTION] Patching completed for vulnerability {self.chosen_vuln} on nodes {self.chosen_node}.")
            self.patching_time_remaining = -1  # Reset patching time for the next cycle

            action = []

            for node in self.chosen_node:

                node_obj = self.env.node_objects[node]

                patching_success = (random.random() <= self.success_probability)

                if getattr(node_obj, "is_off", False):
                # Downed node cannot be patched
                    patching_success = False

                for vuln in node_obj.vulnerabilities:
                    if vuln.cve_id == self.chosen_vuln:
                        vulnerability = vuln
                        break

                if patching_success:

                    # Remove the vulnerability from vulnerabilities_discovered
                    if vulnerability.cve_id in self.vulnerabilities_discovered:
                        if node in self.vulnerabilities_discovered[vulnerability.cve_id]:
                            self.vulnerabilities_discovered[vulnerability.cve_id].remove(node)
                            if not self.vulnerabilities_discovered[vulnerability.cve_id]:  # If no more nodes are affected by this vulnerability
                                del self.vulnerabilities_discovered[vulnerability.cve_id]

                    if vulnerability.cve_id not in self.cve_patched:
                        self.cve_patched.append(vulnerability.cve_id)

                    if (node, vulnerability.cve_id) in self.vulnerability_timestamps:
                        self.vulnerability_timestamps[(node, vulnerability.cve_id)]["patch"] = self.env.current_time

                    self.logger.debug(
                        f"[P_DEFENDER][RESOLUTION] Vulnerability on node {node} successfully patched. "
                    )
                else:
                    self.logger.debug(
                        f"[P_DEFENDER][RESOLUTION] Patch not applied (e.g., zero-day) on node={node}"
                    )

                action.append((node,vulnerability,patching_success))

        return action


    def select_target_vulnerability(self):

        if not self.vulnerabilities_discovered:
            self.logger.debug("[P_DEFENDER][RESOLUTION] No vulnerabilities to resolve.")
            self.chosen_vuln = None
            self.chosen_node = []

        elif self.policy == "importance":
            self.chosen_vuln, self.chosen_node = self.policy_importance()
        elif self.policy == "centrality":
            self.chosen_vuln, self.chosen_node = self.policy_centrality()
        elif self.policy == "severity":
            self.chosen_vuln, self.chosen_node = self.policy_severity()
        elif self.policy == "random":
            self.chosen_vuln, self.chosen_node = self.policy_random()
        else:
            raise ValueError("[P_DEFENDER][SELECT_NODES] Unsupported patch policy")

        if self.chosen_vuln is not None:
            self.patching_time_remaining = self.calculate_patching_time()
            self.env.time_to_patch.append(self.patching_time_remaining)
        else :
            self.patching_time_remaining = -1


    def policy_importance(self):
        """
        Order the vulnerabilities based on the importance of the nodes they affect.
        The importance of a vulnerability is calculated as the average importance of the nodes it affects.
        """
        vuln_importance = {}
        for vuln, nodes in self.vulnerabilities_discovered.items():
            total_importance = sum(self.env.node_objects[node].importance for node in nodes)
            vuln_importance[vuln] = total_importance // len(nodes)

        # Order the vulnerabilities based on their importance
        sorted_vulns = sorted(vuln_importance.items(), key=lambda x: x[1], reverse=True)
        # The most important vulnerability is the first one in the sorted list

        chosen_cve = sorted_vulns[0][0]
        # Get the nodes affected by this vulnerability
        nodes = self.vulnerabilities_discovered[chosen_cve].copy()
        return chosen_cve, nodes


    def policy_centrality(self):
        """
        Order the vulnerabilities based on the centrality of the nodes they affect.
        The centrality of a vulnerability is calculated as the average centrality of the nodes it affects
        """
        vuln_centrality = {}
        for vuln, nodes in self.vulnerabilities_discovered.items():
            total_centrality = sum(self.env.centrality[node] for node in nodes)
            vuln_centrality[vuln] = total_centrality // len(nodes)

        # Order the vulnerabilities based on their centrality
        sorted_vulns = sorted(vuln_centrality.items(), key=lambda x: x[1], reverse=True)
        # The most central vulnerability is the first one in the sorted list
        chosen_cve = sorted_vulns[0][0]
        # Get the nodes affected by this vulnerability
        nodes = self.vulnerabilities_discovered[chosen_cve].copy()
        return chosen_cve, nodes


    def policy_severity(self):
        """
        Order the vulnerabilities based on their CVSS score.
        The severity of a vulnerability is given by its CVSS score.
        """
        vuln_severity = {}
        for vuln, nodes in self.vulnerabilities_discovered.items():
            node_obj = self.env.node_objects[next(iter(nodes))]  # Get the first node affected by this vulnerability
            for v in node_obj.vulnerabilities:
                if v.cve_id == vuln:
                    vulnerability = v
                    break
            vuln_severity[vuln] = vulnerability.cvss_score

        # Order the vulnerabilities based on their severity
        sorted_vulns = sorted(vuln_severity.items(), key=lambda x: x[1], reverse=True)
        # The most severe vulnerability is the first one in the sorted list
        chosen_cve = sorted_vulns[0][0]
        # Get the nodes affected by this vulnerability
        nodes = self.vulnerabilities_discovered[chosen_cve].copy()
        return chosen_cve, nodes


    def policy_random(self):
        """
        Randomly shuffle the vulnerabilities and select the first one.
        """
        vuln = random.choice(list(self.vulnerabilities_discovered.keys()))
        nodes = self.vulnerabilities_discovered[vuln]
        return vuln, nodes


    def calculate_patching_time(self):

        node = next(iter(self.chosen_node))
        node_obj = self.env.node_objects[node]

        time_base = 1  # Base time in days for patching a vulnerability

        # BETA
        beta = sp.DEFENDER_EFFORT

        # GAMMA
        for vuln in node_obj.vulnerabilities:
            if vuln.cve_id == self.chosen_vuln:
                vulnerability = vuln
                break

        node_product = set(vulnerability.products) & set(node_obj.products)
        if any(p in APPLICATION_PRODUCTS for p in node_product):
            # If the vulnerability is for an application, gamma = 1
            gamma = 1
        elif any(p in OS_PRODUCTS for p in node_product):
            # If the vulnerability is for an OS, gamma = 2
            gamma = 2

        # DELTA
        # Delta is the number of vulnerable nodes that need to be patched.
        delta = len(self.chosen_node) * 0.3

        time_to_fix = time_base * beta * gamma * delta

        self.logger.debug(f"[P_DEFENDER][PATCHING_TIME] Calculated patching time for vulnerability {self.chosen_vuln} on nodes {self.chosen_node}: {time_to_fix:.2f} days (alpha={0}, beta={beta}, gamma={gamma}, delta={delta}, omega={0})")

        return math.ceil(time_to_fix)