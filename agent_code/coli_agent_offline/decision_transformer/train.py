# don't confuse with train.py files used for --train; we don't do online RL
import argparse
import pickle
import random
import sys
from time import time

import numpy as np
import torch
from models.decision_transformer import DecisionTransformer

# from decision_transformer.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg
# from decision_transformer.models.decision_transformer import DecisionTransformer
# from decision_transformer.models.mlp_bc import MLPBCModel
# from decision_transformer.training.act_trainer import ActTrainer
# from decision_transformer.training.seq_trainer import SequenceTrainer

# TODO: load last 4 trajectories into one?

# TODO: remove this function once _get_rewards_to_go is implemented
def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        # print(t)
        # print(x[t], discount_cumsum[t + 1])
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
        # print(discount_cumsum[t], '\n')
    return discount_cumsum


def _get_rewards_to_go(x: np.array) -> np.array:
    """
    Returns the rewards to go from a list of rewards.

    Examples:
    [0, 0, 1, 0, 5, 0] --> [6, 6, 6, 5, 5, 0]
    [1, 1, 1, 1, 1, 1] --> [6, 5, 4, 3, 2, 1]
    """
    r = np.zeros_like(x)
    r[0] = sum(x)  # take sum of array as first value
    for i in range(len(r) - 1):
        r[i + 1] = (
            r[i] - x[i]
        )  # compute next value in rewards_to_go by subtracting next val in x from current val in rewards_to_go

    return r


def main(variant):
    device = variant.get("device", "cuda")

    max_ep_len = 401
    state_dim = 49
    action_dim = 6

    # load dataset
    # [[(s, a, r), (s, a, r)], [(s, a, r,), (s, a, r), (s, a, r)]] <-- two trajectories
    trajectories = np.load("trajectories/trajectories_2022-03-22T12:55:28:880375.npy")

    # save trajectories into separate lists
    states, trajectory_lens, returns = [], [], []
    for trajectory in trajectories:  # one trajectory == one episode
        states.append(trajectory[:, 0])
        trajectory_lens.append(len(trajectory[:, 0]))
        returns.append(trajectory[:, 2].sum())

    trajectory_lens, returns = np.array(trajectory_lens), np.array(returns)

    # TODO: input normalization needed?

    num_timesteps = sum(trajectory_lens)
    num_trajectories = len(trajectory_lens)  # we don't do any top-k %, so just take all

    print("=" * 50)
    print(f"{len(trajectory_lens)} trajectories, {num_timesteps} timesteps found")
    print(f"Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}")
    print(f"Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}")
    print("=" * 50)

    K = variant["K"]
    batch_size = variant["batch_size"]  # NOTE: needed for Trainer
    # num_eval_episodes = variant['num_eval_episodes']
    # pct_traj = variant.get('pct_traj', 1.)

    # TODO: is N usually the trajectory length or the number of episodes?
    def get_batch(batch_size=256, max_len=K):  # TODO: why not batch_size=batch_size?

        # get batch_size=256 random trajectory indices
        trajectory_idx = np.random.choice(
            np.arange(num_trajectories), size=batch_size, replace=True
        )

        states, actions, rewards, done_idx, return_to_go, timesteps, mask = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )

        # construct one batch
        for trajectory_index in trajectory_idx:
            trajectory = trajectories[trajectory_index]  # current trajectory

            # get random int in [0, traj_len=10 - 1] ==> get random sequences from trajectory (obeying max_len)
            rng_offset = random.randint(0, trajectory[:, 0].shape[0] - 1)

            # the reshapes are just torch unsqueezes but with numpy, i.e. shape: [2, 6] --> [1, 2, 6]
            # -1 just means inferring the one that's left over
            assert np.expand_dims(
                trajectory[:, 0][rng_offset : rng_offset + max_len], axis=0
            ) == trajectory[:, 0][rng_offset : rng_offset + max_len].reshape(1, -1, state_dim)
            states.append(
                trajectory[:, 0][rng_offset : rng_offset + max_len].reshape(1, -1, state_dim)
            )
            actions.append(
                trajectory[:, 1][rng_offset : rng_offset + max_len].reshape(1, -1, action_dim)
            )
            rewards.append(
                trajectory[:, 2][rng_offset : rng_offset + max_len].reshape(1, -1, 1)
            )  # NOTE: this might have to stay a reshape and not an expand_dim

            # TODO: do we need 'terminals' or 'dones'?
            # "done signal, equal to 1 if playing the corresponding action in the state should terminate the episode"

            timesteps.append(
                np.arange(rng_offset, rng_offset + states[-1].shape[1]).reshape(1, -1)
            )  # e.g. append [[399, 400, 401, 402]]
            timesteps[-1][timesteps[-1] >= max_ep_len] = (
                max_ep_len - 1
            )  # e.g. timesteps [399, 400, 401, 402] ==> [399, 400, 400, 400]

            # TODO: everything beyond line 141
            # returns_to_go

    model = DecisionTransformer(
        state_dim=state_dim,
        action_dim=action_dim,
        max_length=variant,
        max_ep_len=max_ep_len,
        hidden_size=variant["embed_dim"],
        n_layer=variant["n_layer"],
        n_head=variant["n_head"],
        n_inner=4 * variant["embed_dim"],
        activation_function=variant["activation_function"],
        n_positions=1024,
        resid_pdrop=variant["dropout"],
        attn_pdrop=variant["dropout"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="medium"
    )  # medium, medium-replay, medium-expert, expert
    parser.add_argument("--K", type=int, default=20)  # the size of the context window
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=3)
    parser.add_argument("--n_head", type=int, default=1)
    parser.add_argument("--activation_function", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", "-wd", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=10000)
    parser.add_argument("--num_eval_episodes", type=int, default=100)
    parser.add_argument("--max_iters", type=int, default=10)
    parser.add_argument("--num_steps_per_iter", type=int, default=10000)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    print(discount_cumsum(np.array([0, 0, 1, 0, 5, 0]), gamma=1.0))
    print(discount_cumsum(np.array([0, 0, 1, 0, 5, 5]), gamma=1.0))
    print(discount_cumsum(np.array([0, 0, 0, 0, 0, 0]), gamma=1.0))
    print(discount_cumsum(np.array([1, 1, 1, 1, 1, 1]), gamma=1.0))
    print(discount_cumsum(np.array([0, 5, 5, 5, 5, 5, 0]), gamma=1.0))
    print(type(discount_cumsum(np.array([0, 5, 5, 5, 5, 5, 0]), gamma=1.0)))
    print(discount_cumsum(np.array([0, 5, 5, 5, 5, 5, 0]), gamma=1.0).shape)

    # main(variant=vars(args))
    print(_get_rewards_to_go(np.array([0, 0, 1, 0, 5, 0])))
    print(_get_rewards_to_go(np.array([0, 0, 1, 0, 5, 5])))
    print(_get_rewards_to_go(np.array([0, 0, 0, 0, 0, 0])))
    print(_get_rewards_to_go(np.array([1, 1, 1, 1, 1, 1])))
    print(_get_rewards_to_go(np.array([0, 5, 5, 5, 5, 5, 0])))
    print(type(_get_rewards_to_go(np.array([0, 5, 5, 5, 5, 5, 0]))))
    print(_get_rewards_to_go(np.array([0, 5, 5, 5, 5, 5, 0])).shape)
