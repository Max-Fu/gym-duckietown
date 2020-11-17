import ast
import argparse
import logging

import os
import numpy as np

# Duckietown Specific
from reinforcement.pytorch.ddpg import DDPG
from reinforcement.pytorch.rcrl import RCRL
from utils.env import launch_env
from utils.wrappers import NormalizeWrapper, ImgWrapper, DtRewardWrapper, ActionWrapper, ResizeWrapper


def _enjoy(args):
    # Launch the env with our helper function
    env = launch_env()
    print("Initialized environment")

    # Wrappers
    env = ResizeWrapper(env)
    env = NormalizeWrapper(env)
    env = ImgWrapper(env)  # to make the images from 160x120x3 into 3x160x120
    env = ActionWrapper(env)
    env = DtRewardWrapper(env)
    print("Initialized Wrappers")

    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize policy
    if args.rcrl:
        policy = RCRL(state_dim, action_dim, max_action, prior_dim=1, lr_actor=args.lr_actor, lr_critic=args.lr_critic, lr_prior=args.lr_prior)
    else:
        policy = DDPG(state_dim, action_dim, max_action, net_type="cnn")
    if args.rcrl: 
        fn = "rcrl"
    else:
        fn = "ddpg"
    policy.load(filename=fn, directory="reinforcement/pytorch/models/{}/".format(args.folder_hash))

    obs = env.reset()
    done = False

    while True:
        while not done:
            action = policy.predict(np.array(obs))
            # Perform action
            obs, reward, done, _ = env.step(action)
            env.render()
        done = False
        obs = env.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rcrl", action="store_true", default=False)
    parser.add_argument("--folder_hash", required=True, type=str)
    _enjoy(parser.parse_args())
