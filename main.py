import os
import sys
import gym
import yaml
import time
import logging
from datetime import datetime
import random
from shutil import copyfile
import click
from agents.DQN import DQN
from agents.DDQN import DDQN
from agents.DDQN_PER_Prop import DDQN_PER_Prop

logging.basicConfig(level=logging.DEBUG)

@click.command()
@click.argument("config_file")
def main(config_file):
    # Load main config file
    with open(config_file, "r") as f:
        config = yaml.load(f)

    result_path = config["result_dir"]
    agent_type = config["agent"]
    agent_config_file = os.path.join(config["agent_config_dir"], str(agent_type) + ".yml")
    mode = config["mode"]
    environment = config["environment"]
    environment_seed = config["environment_seed"]

    # Load config file for agent
    with open(agent_config_file, "r") as f:
        agent_config = yaml.load(f)

    # Create output directory
    time_str = time.strftime("%Y%m%d_%H%M%S")
    result_path = os.path.join(result_path, agent_type, time_str)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    agent_config["render_environment"] = config["render_environment"]
    agent_config["max_episode"] = config["max_episode"]
    agent_config["max_step"] = config["max_step"]
    agent_config["slide_window"] = config["slide_window"]
    agent_config["result_path"] = result_path

    # Save config files to output directory
    copyfile(config_file, os.path.join(result_path, os.path.basename(config_file)))
    copyfile(config_file, os.path.join(result_path, os.path.basename(agent_config_file)))

    logging.info(mode + " with {} algorithm in environment {}".format(agent_type, environment))
    logging.info("Results will be saved at {}".format(result_path))

    # Initialize environment
    env = gym.make('CartPole-v1')
    env.seed(environment_seed)
    env = env.unwrapped

    # Build/load agent
    if agent_type == "DQN":
        agent = DQN(agent_config, env)
        agent.train()
    elif agent_type == "DDQN":
        agent = DDQN(agent_config, env)
        agent.train()
    elif agent_type == "DDQN_PER_Prop":
        agent = DDQN_PER_Prop(agent_config, env)
        agent.train()
    else:
        raise KeyError("Agent type does not exist")

    # Train or play
    if mode == "train":
        agent.train()
    elif mode == "play":
        agent.play()


if __name__ == '__main__':
    main()