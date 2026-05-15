import os
import logging

from Scenario import SCENARIOS

DISABLE_FILE_LOG = False

scenario_loggers = {}

# Initialized here because training.py imports this reference directly
logger_training_DQNAttacker = logging.getLogger("training_DQN_Attacker")
logger_training_DQNAttacker.setLevel(logging.DEBUG)


def setup_logger(name, log_file, level=logging.DEBUG, enable_file=True):
    logger = logging.getLogger(name)
    logger.handlers.clear()
    if enable_file:
        handler = logging.FileHandler(log_file, mode="w")
        handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] [%(name)s] %(message)s'))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def setup_experiment_loggers(experiment_id):
    log_dir = f"logs/exp_{experiment_id:02d}"
    os.makedirs(log_dir, exist_ok=True)

    scenario_loggers.clear()
    for scenario_id, config in SCENARIOS.items():
        attacker = config["attacker"]
        defender = config["defender"]
        log_file = (
            f"{log_dir}/scenario{scenario_id}_"
            f"{attacker}_attacker_"
            f"{defender if defender else 'no'}_defender.log"
        )
        scenario_loggers[scenario_id] = setup_logger(
            name=f"scenario{scenario_id}",
            log_file=log_file,
            enable_file=not DISABLE_FILE_LOG
        )

    setup_logger(
        name="training_DQN_Attacker",
        log_file=f"{log_dir}/training_DQNAttacker.log",
        enable_file=not DISABLE_FILE_LOG
    )
