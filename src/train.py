from collections import defaultdict
import os
import time
from typing import Tuple, Dict

import gym
import numpy as np
from tqdm import tqdm

from src.ddpg_agent import DDPGAgent
from src.replay import ReplayBuffer


def process_obs(obs):
    if isinstance(obs, dict):
        return np.concatenate([v for k, v in obs.items()])
    return obs


def play_episode(
    env,
    agent,
    replay_buffer,
    train=True,
    explore=True,
    render=False,
    max_steps=200,
    batch_size=64,
):
    ep_data, done = defaultdict(list), False
    obs, _ = env.reset()
    obs = process_obs(obs)

    if render:
        env.render()

    ep_timesteps, ep_return = 0, 0.0
    while not done:
        action = agent.act(obs, explore=explore)
        nobs, reward, done, _, _ = env.step(action)
        nobs = process_obs(nobs)

        if train:
            replay_buffer.push(
                np.array(obs, dtype=np.float32),
                np.array(action, dtype=np.float32),
                np.array(nobs, dtype=np.float32),
                np.array([reward], dtype=np.float32),
                np.array([done], dtype=np.float32),
            )
            if len(replay_buffer) >= batch_size:
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

    return ep_timesteps, ep_return, ep_data


def train_agent(
    env: gym.Env, config, output: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    agent = DDPGAgent(
        action_space=env.action_space, observation_space=env.observation_space, **config
    )
    replay_buffer = ReplayBuffer(config["buffer_capacity"])

    eval_returns_all = []
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
            ep_timesteps, ep_return, ep_data = play_episode(
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
                eval_returns = 0
                for _ in range(config["eval_episodes"]):
                    _, ep_return, _ = play_episode(
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
                if output:
                    pbar.write(
                        f"Evaluation at timestep {timesteps_elapsed} returned a mean returns of {eval_returns}"
                    )
                eval_returns_all.append(eval_returns)
                eval_timesteps_all.append(timesteps_elapsed)
                eval_times_all.append(time.time() - start_time)
                if eval_returns >= config["target_return"]:
                    pbar.write(
                        f"Reached return {eval_returns} >= target return of {config['target_return']}"
                    )
                    break

    if config["save_filename"]:
        print(
            "Saving to: ",
            agent.save(os.path.join("../checkpoints", config["save_filename"])),
        )

    return (
        np.array(eval_returns_all),
        np.array(eval_timesteps_all),
        np.array(eval_times_all),
        run_data,
    )
