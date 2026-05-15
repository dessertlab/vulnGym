# NetworkNode.py
import os
import json
import random
import simulation_parameters as sp

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from macro import *

class NetworkNode:
    """
    Class representing a network node.

    A node contains:
    - structural information (zone, product, importance)
    - vulnerabilities
    - dynamic state related to attacker/defender interaction
    """

    def __init__(self, node_id, name, zone, importance, products, phishing=False):

        # =========================
        # Structural attributes
        # =========================

        # Unique identifier of the node
        self.node_id = node_id

        # Human-readable node name
        self.name = name

        # Network zone (external, dmz, internal, database)
        self.zone = zone

        # Importance score (used for reward or prioritization)
        self.importance = importance

        # List of vulnerabilities associated with this node
        self.vulnerabilities = []

        # Current security state of the node
        self.state = NODE_SAFE

        # Flag used for traversal/algorithms (e.g., graph search)
        self.visited = False

        # Software products installed on the node
        self.products = products

        # Whether phishing is possible on this node
        self.phishing = phishing


        # =========================
        # Attack simulation attributes
        # =========================

        # Whether the attacker has scanned the node
        self.scanned = False

        # Whether the node is attackable (None = unknown before scan)
        self.attackable = None

        # Whether the attacker has initial access
        self.attacker_has_access = False

        # Whether the attacker has elevated privileges
        self.attacker_has_privileges = False

        # Whether the attacker has established persistence
        self.attacker_persistent = False

        # Whether data has been exfiltrated from this node
        self.data_exfiltrated = False

        # Whether credentials have been stolen from this node
        self.credentials_stolen = False

        # Node is unavailable due to DoS
        self.is_off = False

        # Node has been wiped (destructive attack)
        self.is_wiped = False


    def reset(self):
        """
        Reset the node to its initial state for a new simulation episode.
        """

        # Reset security state
        self.state = NODE_SAFE

        # Reset traversal flags
        self.visited = False

        # Reset attacker-related flags
        self.attacker_has_access = False
        self.attacker_has_privileges = False
        self.attacker_persistent = False

        # Reset impact flags
        self.is_off = False
        self.is_wiped = False

        # Reset attack progress
        self.data_exfiltrated = False
        self.credentials_stolen = False
        self.scanned = False
        self.attackable = None

        # Optional attribute (used by defender logic)
        self.unpatchable = False

        # Reset vulnerabilities
        for vuln in self.vulnerabilities:
            vuln.reset()

        # Clear vulnerability list (will be regenerated)
        self.vulnerabilities.clear()


def create_node(node_id, zone, importance):
    """
    Create a NetworkNode instance for a given zone.

    The function:
    - assigns a product randomly from the zone-specific list
    - determines if phishing is possible for that product

    Args:
        node_id (int): Unique node identifier
        zone (str): Network zone (e.g. "dmz", "internal")
        importance (float): Node importance score

    Returns:
        NetworkNode: Initialized node object
    """

    # Generate node name
    name = f"{zone}_node_{node_id}"

    with open(os.path.join(_ROOT, "config", "products.json"), "r") as f:
        products_data = json.load(f)

    # Get products for a specific zone

    products_by_zone=[product for product, info in products_data.items() if zone in info["zones"]]

    # Assign products depending on the zone
    if zone == "database":
        # Database nodes have only one product
        products = [random.choice(sp.apt_database_products)]
    else:
        # Other zones can have multiple products
        products = random.sample(products_by_zone, k=sp.NUM_PRODUCTS_PER_NODE)

    phishing = any(products_data[product]["phishing"] for product in products_data)

    return NetworkNode(node_id, name, zone, importance, products, phishing)


def recompute_state(node):
    """
    Recompute the node security state based on its current conditions.

    Rules:
    - If the node is turned off (DoS), do not update the state
    - If attacker has persistence → DANGER
    - If there are unpatched vulnerabilities → DANGER
    - Otherwise → SAFE

    """

    # Do not update state if node is offline
    if getattr(node, "is_off", False):
        return

    # Persistent attacker presence overrides everything
    if getattr(node, "attacker_persistent", False):
        node.state = NODE_DANGER
        return

    # Check for unpatched vulnerabilities
    has_unpatched = node_has_unpatched_vulns(node)

    if has_unpatched:
        node.state = NODE_DANGER
    else:
        node.state = NODE_SAFE


def node_has_unpatched_vulns(node_obj):
    """
    Check if a node has at least one unpatched vulnerability.

    Args:
        node_obj (NetworkNode): Node to check

    Returns:
        bool: True if at least one vulnerability is not patched
    """

    return any(
        not getattr(v, "patched", False)
        for v in getattr(node_obj, "vulnerabilities", [])
    )