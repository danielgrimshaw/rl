import random
from typing import Any, Iterable, Optional, Tuple

import gym
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
from tqdm import tqdm

import gridworld  # noqa: F401


class Network(keras.Model):
    def __init__(self, action_dim: int = 5, input_shape: Tuple = (1, 8, 8, 1)) -> None:
        super().__init__()
        self.entry = keras.layers.Conv2D(
            8,
            2,
            padding="same",
            input_shape=input_shape,
            activation="relu",
            kernel_initializer="random_normal",
        )
        self.reshape = keras.layers.Flatten()
        self.dense = keras.layers.Dense(
            32, kernel_initializer="random_normal", activation="relu"
        )
        self.logits = keras.layers.Dense(action_dim, kernel_initializer="random_normal")

    def call(self, inputs: np.ndarray) -> tf.Tensor:
        x = tf.reshape(tf.cast(inputs, tf.float16), (-1, 8, 8, 1))
        return self.logits(self.dense(self.reshape(self.entry(x))))

    def process(self, observations: np.ndarray) -> tf.Tensor:
        return self.predict_on_batch(observations)


class Agent:
    def __init__(
        self, action_dim: int = 5, input_shape: Tuple[int, int] = (1, 8 * 8)
    ) -> None:
        self.network = Network(action_dim, input_shape)
        self.network.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
            run_eagerly=True,
        )
        self.policy = self.policy_mlp

    def policy_mlp(self, observations: np.ndarray) -> tf.Tensor:
        observations = observations.reshape(1, -1)
        action_logits = self.network.process(observations)
        return tf.random.categorical(tf.math.log(action_logits), num_samples=1)

    def get_action(self, observations: np.ndarray) -> tf.Tensor:
        return self.policy(observations)

    def learn(self, obs: np.ndarray, actions: np.ndarray, **kwargs: Any) -> None:
        self.network.fit(obs, actions, **kwargs)


def evaluate(
    agent: Agent, env: gym.Env, render: bool = True
) -> Tuple[int, float, bool, Optional[Any]]:
    obs, ep_reward, done, step_num, info = env.reset(), 0.0, False, 0, None
    while not done:
        action = agent.get_action(obs)
        obs, reward, done, info = env.step(action)
        ep_reward += reward
        step_num += 1
        if render:
            env.render()
    return step_num, ep_reward, done, info


def play_one(
    agent: Agent, env: gym.Env, jitter: float = 0.0, render: bool = False
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Plays a full episode and returns with what happened"""
    obs, ep_reward, done, step_num = env.reset(), 0.0, False, 0
    observations, actions = [], []
    while not done:
        action = agent.get_action(obs)
        if random.random() < jitter:
            action = env.action_space.sample()
        observations.append(np.array(obs.reshape(-1)))
        actions.append(np.squeeze(action, 0))
        obs, reward, done, info = env.step(action)
        ep_reward += reward
        step_num += 1
        if render:
            env.render()
    env.close()
    return observations, actions, ep_reward


def get_best(
    games: Iterable[Tuple[np.ndarray, np.ndarray, float]], cutoff_percentile: int
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    game_obs, game_acts, game_rewards = zip(*games)
    thresh = np.percentile(game_rewards, cutoff_percentile)
    indices = [i for i, reward in enumerate(game_rewards) if reward >= thresh]
    print(f"Training with {len(indices)} best of {len(game_rewards)} examples...")

    best_obs, best_acts = [game_obs[i] for i in indices], [
        game_acts[i] for i in indices
    ]
    return (
        np.array([a for b in best_obs for a in b]),
        np.array([a for b in best_acts for a in b]),
        np.mean(game_rewards),
        thresh,
    )


def get_action_dist(action_index: int, action_dim: int = 5) -> np.ndarray:
    """one hot the action"""
    action_dist = np.zeros(action_dim).astype(int)
    action_dist[action_index] = 1
    return action_dist


def train(
    env_id: str = "gridworld.gridworld:Gridworld-v0",
    num_games: int = 100,
    train_percentile: int = 95,
    num_epochs: int = 10,
) -> Agent:
    env = gym.make(env_id)
    agent = Agent(env.action_space.n, env.obs_space.shape)

    mean_rewards = []
    thresholds = []
    for i in tqdm(range(num_epochs)):
        games = [
            play_one(agent, env)  # , jitter=1 if i < num_epochs // 10 else 0
            for _ in range(num_games)
        ]
        best_obs, best_acts, mean_reward, threshold = get_best(games, train_percentile)
        action_dists = np.array([get_action_dist(a.item()) for a in best_acts])
        agent.learn(
            best_obs.astype("float16"),
            action_dists.astype("float16"),
            batch_size=128,
            epochs=10,
        )
        mean_rewards.append(mean_reward)
        thresholds.append(threshold)
        print(f"Episode {i + 1} threshold={threshold}, average={mean_reward}")

    plt.plot(mean_rewards, "r-", label="mean reward")
    plt.plot(thresholds, "g--", label="threshold reward")
    plt.legend()
    plt.grid()
    plt.show()
    # evaluate(agent, env, render=True)
    return agent


if __name__ == "__main__":
    train()
