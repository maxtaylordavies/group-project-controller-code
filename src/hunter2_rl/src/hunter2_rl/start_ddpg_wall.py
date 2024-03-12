#!/usr/bin/env python3 

import gym
from gym import wrappers
import numpy as np
from tqdm import tqdm
# ROS packages required
import rospy
import rospkg
# import our training environment
from hunter2_rl.task_envs.hunter2_maze import Hunter2Maze

from collections import defaultdict
import os
import time
from typing import Tuple, Dict

from hunter2_rl.replay import ReplayBuffer
from hunter2_rl.ddpg_agent import DDPGAgent



def process_obs(obs):
    if isinstance(obs, dict):
        return np.concatenate([v for k, v in obs.items()])
    return obs


def play_episode(
    env,
    agent,
    replay_buffer,
    train=True,
    save_to_buffer=True,
    explore=True,
    render=False,
    record_fp="",
    max_steps=500,
    batch_size=64,
):
    ep_data, done = defaultdict(list), False
    obs, _ = env.reset()
    obs = process_obs(obs)

    if render:
        env.render()

    if record_fp != "":
        env.init_recording(record_fp)

    ep_timesteps, ep_return, success = 0, 0.0, False
    while not done:
        action = agent.act(obs, explore=explore)
        nobs, reward, done, _, _ = env.step(action)
        nobs = process_obs(nobs)

        # [TODO] - Change this to fit our reward function
        if reward >= 0:
            success = True

        if train or save_to_buffer:
            replay_buffer.push(
                np.array(obs, dtype=np.float32),
                np.array(action, dtype=np.float32),
                np.array(nobs, dtype=np.float32),
                np.array([reward], dtype=np.float32),
                np.array([done], dtype=np.float32),
            )
        if train and len(replay_buffer) >= batch_size:
            batch = replay_buffer.sample(batch_size)
            new_data = agent.update(batch)
            for k, v in new_data.items():
                ep_data[k].append(v)

        ep_timesteps += 1
        ep_return += reward

        if render:
            env.render()
        if max_steps == ep_timesteps:
            break
        obs = nobs

    if record_fp != "":
        env.finish_recording()

    return ep_timesteps, ep_return, ep_data, success


def train_agent(
    env: gym.Env, config, output: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    agent = DDPGAgent(
        action_space=env.action_space, observation_space=env.observation_space, **config
    )
    replay_buffer = ReplayBuffer(config["buffer_capacity"])

    eval_returns_all = []
    eval_success_rates_all = []
    eval_timesteps_all = []
    eval_times_all = []
    run_data = defaultdict(list)

    timesteps_elapsed, start_time = 0, time.time()
    with tqdm(total=config["max_timesteps"]) as pbar:
        while timesteps_elapsed < config["max_timesteps"]:
            # if we've reached max training duration, finish
            seconds_elapsed = time.time() - start_time
            if seconds_elapsed > config["max_time"]:
                pbar.write(f"Training ended after {seconds_elapsed}s.")
                break

            # perform any time-dependent hyperparam updates
            agent.schedule_hyperparameters(timesteps_elapsed, config["max_timesteps"])

            # sample episode
            ep_timesteps, ep_return, ep_data, _ = play_episode(
                env,
                agent,
                replay_buffer,
                train=True,
                explore=True,
                render=False,
                max_steps=config["episode_length"],
                batch_size=config["batch_size"],
            )

            # update progress bar
            timesteps_elapsed += ep_timesteps
            pbar.update(ep_timesteps)

            # store data from episode
            for k, v in ep_data.items():
                run_data[k].extend(v)
            run_data["train_ep_returns"].append(ep_return)

            # possibly run evaluation
            if timesteps_elapsed % config["eval_freq"] < ep_timesteps:
                eval_returns, success_rate = 0, 0
                for _ in range(config["eval_episodes"]):
                    _, ep_return, _, success = play_episode(
                        env,
                        agent,
                        replay_buffer,
                        train=False,
                        explore=False,
                        render=False,
                        max_steps=config["episode_length"],
                        batch_size=config["batch_size"],
                    )
                    eval_returns += ep_return / config["eval_episodes"]
                    success_rate += int(success) / config["eval_episodes"]
                if output:
                    pbar.write(
                        f"Evaluation at timestep {timesteps_elapsed}: mean return {eval_returns}, success rate {round(100 * success_rate)}%"
                    )
                eval_returns_all.append(eval_returns)
                eval_success_rates_all.append(success_rate)
                eval_timesteps_all.append(timesteps_elapsed)
                eval_times_all.append(time.time() - start_time)

                if eval_returns == max(eval_returns_all):
                    agent.save(os.path.join("../checkpoints/best_return.pt"))
                if success_rate == max(eval_success_rates_all):
                    agent.save(os.path.join("../checkpoints/best_success.pt"))
                if success_rate > 0.99:
                    print("Success rate reached 100%, stopping training.")
                    break

    print(
        "Saving to: ",
        agent.save(os.path.join("../checkpoints/latest.pt")),
    )

    return (
        np.array(eval_returns_all),
        np.array(eval_timesteps_all),
        np.array(eval_times_all),
        run_data,
    )



if __name__ == '__main__':

    rospy.init_node('example_hunter2_maze_ddpg', anonymous=True, log_level=rospy.WARN)

    # Create the Gym environment
    env = gym.make('Hunter2Maze-v0')
    rospy.loginfo("Gym environment done")

    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('hunter2_rl')
    outdir = pkg_path + '/training_results'
    # env = wrappers.Monitor(env, outdir, force=True)
    # rospy.loginfo("Monitor Wrapper started")

    last_time_steps = np.ndarray(0)

    config = {}

    # Loads parameters from the ROS param server
    # Parameters are stored in a yaml file inside the config directory
    # They are loaded at runtime by the launch file
    action_space = env.action_space
    observation_space = env.observation_space

    config['gamma'] = rospy.get_param("/hunter2/gamma")
    config['tau'] = rospy.get_param("/hunter2/tau")

    config['critic_learning_rate'] = rospy.get_param("/hunter2/critic_learning_rate")
    config['policy_learning_rate'] = rospy.get_param("/hunter2/policy_learning_rate")
    config['critic_hidden_size'] = rospy.get_param("/hunter2/critic_hidden_size")
    config['policy_hidden_size'] = rospy.get_param("/hunter2/policy_hidden_size")

    config['max_timesteps'] = rospy.get_param("/hunter2/max_timesteps")
    config['max_time'] = rospy.get_param("/hunter2/max_time")
    config['eval_freq'] = rospy.get_param("/hunter2/eval_freq")
    config['eval_episodes'] = rospy.get_param("/hunter2/eval_episodes")
    config['batch_size'] = rospy.get_param("/hunter2/batch_size")
    config['buffer_capacity'] = rospy.get_param("/hunter2/buffer_capacity")
    config['episode_length'] = rospy.get_param("/hunter2/episode_length")
   



    # Initialises the algorithm that we are going to use for learning
    ddqg = DDPGAgent(
                    observation_space,
                    action_space,
                    **config
                    )

    train_agent(env, config)

    env.close()



