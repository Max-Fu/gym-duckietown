import ast
import argparse
import logging

import os
import numpy as np

# Duckietown Specific
from reinforcement.pytorch.ddpg import DDPG
from reinforcement.pytorch.rcrl import RCRL
from reinforcement.pytorch.utils import seed, evaluate_policy, ReplayBuffer
from utils.env import launch_env
from utils.wrappers import NormalizeWrapper, ImgWrapper, DtRewardWrapper, ActionWrapper, ResizeWrapper
from gym_duckietown.simulator import AGENT_SAFETY_RAD
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

POSITION_THRESHOLD = 0.04
REF_VELOCITY = 0.7
FOLLOWING_DISTANCE = 0.24
AGENT_SAFETY_GAIN = 1.15

def _train(args):
    if not os.path.exists("./results"):
        os.makedirs("./results")
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    # Launch the env with our helper function
    # update args to include different towns
    env = launch_env('Duckietown-loop_pedestrians-v0')
    print("Initialized environment")

    # Wrappers
    env = ResizeWrapper(env)
    env = NormalizeWrapper(env)
    env = ImgWrapper(env)  # to make the images from 160x120x3 into 3x160x120
    env = ActionWrapper(env)
    env = DtRewardWrapper(env)
    print("Initialized Wrappers")

    # Set seeds
    seed(args.seed)

    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize policy
    # TODO: choose policy to be DDPG or RCRL, net_type
    if args.rcrl:
        # the net_type of RCRL is fixed to "cnn", prior is assigned to be the exp(-alpha*k) 
        policy = RCRL(state_dim, action_dim, max_action, prior_dim=1)
    else:
        policy = DDPG(state_dim, action_dim, max_action, net_type="cnn")
    replay_buffer = ReplayBuffer(args.replay_buffer_max_size, additional=args.rcrl)
    print("Initialized Model")

    # Evaluate untrained policy
    evaluations = [evaluate_policy(env, policy)]

    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True
    episode_reward = None
    env_counter = 0
    reward = 0
    episode_timesteps = 0
    last_sample = None
    writer = SummaryWriter(log_dir=os.path.join(args.model_dir, 'log_tb'))
    if args.rcrl: 
        fn = "rcrl"
    else:
        fn = "ddpg"
    print("Starting training")
    while total_timesteps < args.max_timesteps:

        print("timestep: {} | reward: {}".format(total_timesteps, reward))

        if done:
            if total_timesteps != 0:
                print(
                    ("Total T: %d Episode Num: %d Episode T: %d Reward: %f")
                    % (total_timesteps, episode_num, episode_timesteps, episode_reward)
                )
                losses = policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau)
                # Write losses to tensorboard 
                for tag, val in losses.items():
                    writer.add_scalar('loss/'+tag, val, total_timesteps)
                
                # Evaluate episode
                if timesteps_since_eval >= args.eval_freq:
                    timesteps_since_eval %= args.eval_freq
                    evaluations.append(evaluate_policy(env, policy))
                    print("rewards at time {}: {}".format(total_timesteps, evaluations[-1]))
                    # Write rewards to tensorboard 
                    writer.add_scalar('rewards', evaluations[-1], total_timesteps)

                    if args.save_models:
                        policy.save(file_name=fn, directory=args.model_dir)
                    np.savez("./results/rewards.npz", evaluations)

            # Reset environment
            env_counter += 1
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_num += 1

        # Select action randomly or according to policy
        if total_timesteps < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = policy.predict(np.array(obs))
            if args.expl_noise != 0:
                action = (action + np.random.normal(0, args.expl_noise, size=env.action_space.shape[0])).clip(
                    env.action_space.low, env.action_space.high
                )

        # Perform action
        new_obs, reward, done, _ = env.step(action)

        if episode_timesteps >= args.env_timesteps:
            done = True

        done_bool = 0 if episode_timesteps + 1 == args.env_timesteps else float(done)
        episode_reward += reward

        if args.rcrl: 
            # augment state input (obs, risk to closest object)
            current_world_objects = env.objects
            obj_distances = []
            for obj in current_world_objects:
                if not obj.static:
                    obj_safe_dist = abs(
                        obj.proximity(env.cur_pos, AGENT_SAFETY_RAD * AGENT_SAFETY_GAIN, true_safety_dist=True)
                    )
                    obj_distances.append(obj_safe_dist)
            min_dist = min(obj_distances)
            
            # reduce variance by using exponential decay
            exp_neg_min_dist = np.exp(-args.dist_param * min_dist)
            
            # Delay 1 step and store data in replay buffer; want one step look ahead
            if last_sample:
                last_sample[-1] = np.array([last_sample[-1], exp_neg_min_dist])
                replay_buffer.add(*last_sample)
            last_sample = [obs, new_obs, action, reward, done_bool, exp_neg_min_dist]
        else: 
            # Store data in replay buffer
            replay_buffer.add(obs, new_obs, action, reward, done_bool)

        obs = new_obs

        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1

    print("Training done, about to save..")
    policy.save(filename=fn, directory=args.model_dir)
    print("Finished saving..should return now!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # DDPG Args
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument(
        "--start_timesteps", default=1e4, type=int
    )  # How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=5e3, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=float)  # Max time steps to run environment for
    parser.add_argument("--save_models", action="store_true", default=True)  # Whether or not models are saved
    parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=32, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument(
        "--policy_noise", default=0.2, type=float
    )  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--env_timesteps", default=500, type=int)  # Frequency of delayed policy updates
    parser.add_argument(
        "--replay_buffer_max_size", default=10000, type=int
    )  # Maximum number of steps to keep in the replay buffer
    parser.add_argument("--model-dir", type=str, default="reinforcement/pytorch/models/")
    parser.add_argument("--rcrl", action="store_true", default=False)
    parser.add_argument("--dist_param", default=1.0, type=float) # when calculating possibility of collision, uses exp(-alpha * k)
    _train(parser.parse_args())
