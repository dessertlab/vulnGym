# =========================
# Node States
# =========================
# Possible states of a node in the network

NODE_SAFE = "safe"        # Node is secure and not compromised
NODE_DANGER = "danger"    # Node is vulnerable or at risk
NODE_PATCHED = "patched"  # Node has been patched (secured against known vulnerabilities)


# =========================
# Campaign Types
# =========================
# High-level attacker objectives

IMPACT_CAMPAIGN = "impact"   # Destructive or high-impact attacks (e.g., wiper, exfiltration)
DOS_CAMPAIGN = "dos"         # Denial of Service attacks


# =========================
# Actions
# =========================
# Possible actions that the attacker can perform

NO_ACTION = "NONE"                             # No operation

SCAN = "SCAN"                                  # Discover vulnerabilities or network information
EXPLOIT = "EXPLOIT"                            # Exploit a vulnerability
WAIT = "WAIT"                                  # Do nothing (pass time)
LATERAL_MOVEMENT = "LATERAL_MOVEMENT"          # Move laterally within the network
CREDENTIALS_THEFT = "CREDENTIALS_THEFT"        # Steal credentials for lateral movement
PRIVILEGE_ESCALATION = "PRIVILEGE_ESCALATION"  # Gain higher privileges
PERSISTENCE = "PERSISTENCE"                    # Maintain long-term access
EXFILTRATION = "EXFILTRATION"                  # Steal data (main goal in impact campaigns)
WIPER = "WIPER"                                # Destroy data or systems

REMOTE_CONTROL = "remote_control"              # Gain remote control over a node/system
DOS = "dos"                                    # Perform Denial of Service attack

ACTION_TYPES = [
    SCAN,
    EXPLOIT,
    LATERAL_MOVEMENT,
    CREDENTIALS_THEFT,
    PRIVILEGE_ESCALATION,
    PERSISTENCE,
    EXFILTRATION,
    WIPER,
    DOS
]
# =========================
# Action Tuple Indexing
# =========================
# Used to access elements in action representations (e.g., tuples)

NODE = 0           # Index of the target node
ACTION = 1         # Index of the action type
VULNERABILITY = 2  # Index of the vulnerability involved


# =========================
# Product Categories
# =========================
# Grouping of software products used to:
# - assign vulnerabilities
# - model attack surfaces
# - define node types

DATABASE_PRODUCTS = [
    # Database systems (high-value targets for data exfiltration)
    "mariadb:mariadb",
    "microsoft:sql",
    "mongodb:mongodb",
    "oracle:oracle",
    "postgresql:postgresql",
    "sqlite:sqlite"
]

SOFTWARE_PRODUCTS = [
    # General software and services (broad attack surface)
    "1password:1password",
    "adobe:adobe",
    "apache:apache",
    "auth0:auth0",
    "avast:avast",
    "bitdefender:bitdefender",
    "debian:debian",
    "microsoft:active_directory",
    "microsoft:defender",
    "microsoft:edge",
    "microsoft:exchange_server",
    "microsoft:office",
    "microsoft:powershell",
    "microsoft:remote",
    "microsoft:windows",
    "microsoft:windows_server",
    "redhat:redhat",
    "vmware:vmware"
]

APPLICATION_PRODUCTS = [
    # Applications and services (used in mixed environments)
    "1password:1password",
    "adobe:adobe",
    "apache:apache",
    "auth0:auth0",
    "avast:avast",
    "bitdefender:bitdefender",
    "mariadb:mariadb",
    "microsoft:active_directory",
    "microsoft:defender",
    "microsoft:edge",
    "microsoft:exchange_server",
    "microsoft:office",
    "microsoft:powershell",
    "microsoft:remote",
    "microsoft:sql",
    "mongodb:mongodb",
    "oracle:oracle",
    "postgresql:postgresql",
    "sqlite:sqlite",
    "vmware:vmware"
]

OS_PRODUCTS = [
    # Operating systems (base layer of each node)
    "debian:debian",
    "redhat:redhat",
    "microsoft:windows",
    "microsoft:windows_server"
]