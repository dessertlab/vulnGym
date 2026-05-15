# =========================
# Action-based Rewards
# =========================

# Reward for forced movement (e.g., lateral movement enforced by environment)
REWARD_FORCED_MOVE = 10


# =========================
# Post-Action Importance Weights
# =========================
# These values represent how important each action is
# depending on the attacker's objective.
# They can be used to scale rewards or guide learning.

ACTION_IMPORTANCE = {

    # Goal: Exfiltrate sensitive data
    "data_exfiltration": {
        "WAIT": 0.0,                   # No progress
        "SCAN": 0.1,                   # Reconnaissance
        "EXPLOIT": 1.0,                # Initial compromise
        "LATERAL_MOVEMENT": 1.0,       # Move towards data
        "CREDENTIALS_THEFT": 1.5,      # Lateral movement enabler
        "PRIVILEGE_ESCALATION": 1.5,   # Gain higher privileges
        "PERSISTENCE": 1.5,            # Maintain access
        "DOS": 0.0,                    # Secondary impact
        "EXFILTRATION": 3.0,           # Main objective
        "WIPER": 3.0,                  # High impact action
    },

    # Goal: Denial of Service on database nodes
    "db_denial_of_service": {
        "WAIT": 0.0,                   # No progress
        "SCAN": 0.1,                   # Reconnaissance
        "EXPLOIT": 1.0,                # Initial compromise
        "LATERAL_MOVEMENT": 1.0,       # Move towards data
        "CREDENTIALS_THEFT": 1.5,      # Lateral movement enabler
        "PRIVILEGE_ESCALATION": 1.5,   # Gain higher privileges
        "PERSISTENCE": 1.5,            # Maintain access
        "DOS": 1.0,                    # Main objective
        "EXFILTRATION": 1.5,           # Secondary relevance
        "WIPER": 1.5,                  # Secondary impact
    }
}


# =========================
# Episode Termination Rewards
# =========================

# Reward when the attacker successfully reaches the goal
REWARD_DONE = 100

# Reward (or penalty) when time limit is exceeded
REWARD_TIME_LIMIT_EXCEEDED = 100


# =========================
# Defender Rewards
# =========================

# Reward when a patch is successfully applied
REWARD_PATCH_SUCCESS = 3

# Penalty when a patch action fails
REWARD_PATCH_FAILURE = -3

# Extra reward when patching an already exploited vulnerability
REWARD_PATCH_ON_EXPLOITED_VULN = 7

# Penalty when there are no vulnerabilities left to patch
REWARD_NO_PATCHABLE_NODE = -3

# Penalty when a new node gets compromised
REWARD_NEW_COMPROMISED_NODE = -10