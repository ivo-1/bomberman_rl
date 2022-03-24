# don't confuse with train.py files used for --train; we don't do online RL
import argparse
import glob
import os
import random
from datetime import datetime

import numpy as np
import torch
from models.decision_transformer import DecisionTransformer
from torch.nn import functional as F
from trainer import Trainer

# from decision_transformer.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg


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

    # load dataset by combining trajectory files (every agent has its own
    # trajectory file that is generated during training) --> multiple files

    # find all trajectories in trajectories folder
    list_of_trajectories = glob.glob("trajectories/*.npy")
    # list_of_trajectories.sort(key=os.path.getctime) # sort by creation time

    print(f"Found the following trajectory files: {list_of_trajectories}")
    agent_trajectories = []
    for agent_trajectory in list_of_trajectories:
        agent_trajectories.append(np.load(agent_trajectory, allow_pickle=True))

    trajectories = np.concatenate(agent_trajectories)

    # save trajectories into separate lists
    states, trajectory_lens, returns = [], [], []
    for trajectory in trajectories:  # one trajectory == one episode
        states.append(trajectory[:, 0])
        trajectory_lens.append(len(trajectory[:, 0]))
        returns.append(trajectory[:, 2].sum())

    trajectory_lens, returns = np.array(trajectory_lens), np.array(returns)

    # for input normalization (later)
    # states = np.concatenate(states, axis=0)
    # state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

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

    def get_batch(batch_size=batch_size, max_len=K):

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
            trajectory = trajectories[
                trajectory_index
            ]  # current trajectory: [length_of_trajectory, 3]

            # get random int in [0, traj_len=10 - 1] ==> get random sequences from trajectory (obeying max_len)
            rng_offset = random.randint(0, trajectory[:, 0].shape[0] - 1)

            # the reshapes are just torch unsqueezes but with numpy, i.e. shape: [2, 6] --> [1, 2, 6]
            # -1 just means inferring the one that's left over
            # assert np.expand_dims(
            #     trajectory[:, 0][rng_offset : rng_offset + max_len], axis=0
            # ) == trajectory[:, 0][rng_offset : rng_offset + max_len].reshape(1, -1, state_dim)
            states.append(
                np.vstack(trajectory[:, 0][rng_offset : rng_offset + max_len]).reshape(
                    1, -1, state_dim
                )  # (1, length_of_trajectory chosen, 49)
            )
            actions.append(
                np.vstack(trajectory[:, 1][rng_offset : rng_offset + max_len]).reshape(
                    1, -1, action_dim
                )
            )

            # NOTE: no vstack needed here and MUST be converted to int!
            rewards.append(
                trajectory[:, 2][rng_offset : rng_offset + max_len].astype(int).reshape(1, -1, 1)
            )  # NOTE: this might have to stay a reshape and not an expand_dim

            # NOTE: do we need 'terminals' or 'dones'?
            # "done signal, equal to 1 if playing the corresponding action in the state should terminate the episode"
            # ANSWER: they don't use the 'dones' that are returned by this function --> no need

            timesteps.append(
                np.arange(rng_offset, rng_offset + states[-1].shape[1]).reshape(
                    1, -1
                )  # [1, length_of_trajectory]
            )  # e.g. append [[399, 400, 401, 402]]
            timesteps[-1][timesteps[-1] >= max_ep_len] = (
                max_ep_len - 1
            )  # e.g. timesteps [[399, 400, 401, 402]] ==> [[399, 400, 400, 400]] when max_ep_len = 401 (1, length_of_trajectory)

            # all future rewards (even over max_len) so that the calculation of returns to go is correct
            rewards_to_go_from_offset = _get_rewards_to_go(
                trajectory[:, 2][rng_offset:]
            )  # (length_of_trajectory from cutoff til end,)

            # append only max_len allowed sequence
            # NOTE: must be cast!
            return_to_go.append(
                rewards_to_go_from_offset[: states[-1].shape[1] + 1].astype(int).reshape(1, -1, 1)
            )  # (1, length_of_trajectory, 1)

            # there is exactly one more state seen in the current sequence than rewards in the current sequence
            # --> add one zero reward to go
            # this happens when the current sampled sequence goes "over" the length of the trajectory its sampled from
            # e.g. with max_len = 10
            # trajectory has timesteps: [..., 128, 129]
            # rng_offset happens to be: 125
            # --> timesteps: 125, 126, ..., 134 are looked at (because max_len = 10) but of course
            # there are no timesteps beyond 129 (later on this is padded)
            if return_to_go[-1].shape[1] <= states[-1].shape[1]:
                return_to_go[-1] = np.concatenate(
                    [return_to_go[-1], np.zeros((1, 1, 1), dtype=int)], axis=1
                )

            sequence_len = states[-1].shape[1]

            # left-padding with zero states if our sequence is shorter than max_len
            # this happens when the offset is high and the rest of the trajectory till episode
            # finish is shorter than max_len
            # --> probably safer to not encode any field state with 0, but rather start with one
            states[-1] = np.concatenate(
                [np.zeros((1, max_len - sequence_len, state_dim), dtype=int), states[-1]], axis=1
            )
            # TODO: normalization  - skip for now

            # left-padding with dummy action (not one-hot encoded, but just whole vector -10)
            actions[-1] = np.concatenate(
                [np.ones((1, max_len - sequence_len, action_dim)) * -10, actions[-1]], axis=1
            )

            # left-padding reward with 0 rewards
            rewards[-1] = np.concatenate(
                [np.zeros((1, max_len - sequence_len, 1)), rewards[-1]], axis=1
            )

            # NOTE: left-padding done/terminals - skip for now as they don't use done_idx

            # left-padding rewards_to_go with zero reward-to-go
            # authors divide the returns to go with `scale` -- as per issue #32 in their repo:
            # > "scale is a normalization hyperparmeter, coarsely chosen so that the rewards would
            # > fall somewhere in the range 0-10.
            #
            # as the max score in Bomberman is (3*5)+9 = 24
            # we will divide by 2.5 per default
            return_to_go[-1] = (
                np.concatenate([np.zeros((1, max_len - sequence_len, 1)), return_to_go[-1]], axis=1)
                / variant["scale"]
            )

            # left-padding timesteps with zeros
            timesteps[-1] = np.concatenate(
                [np.zeros((1, max_len - sequence_len)), timesteps[-1]], axis=1
            )

            # masking the fake tokens (the left-paddings), i.e. [0, 0, 1, 1, 1] if max_len-sequence_len == 2
            mask.append(
                np.concatenate(
                    [np.zeros((1, max_len - sequence_len)), np.ones((1, sequence_len))], axis=1
                )
            )

        # print(len(states)) # batch_size because we have batch_size sequences in the batch
        # print(states[0].shape) # (1, max_len, state_dim)
        # print(states[0].dtype) # int64

        # concatenate all the stuff of all the sequences in the batch to be behind each other and prepare them to be sent to torch
        states = torch.from_numpy(np.concatenate(states, axis=0)).to(
            dtype=torch.float32, device=device
        )

        actions = torch.from_numpy(np.concatenate(actions, axis=0)).to(
            dtype=torch.float32, device=device
        )

        # print(len(rewards))
        # print(rewards[0].shape) # (1, 50, 1) == (1, max_len, reward_dim (i.e. it's a number --> 1))
        # print(rewards[0].dtype)

        rewards = torch.from_numpy(np.concatenate(rewards, axis=0)).to(
            dtype=torch.float32, device=device
        )
        # done_idx = torch.from_numpy(np.concatenate(done_idx, axis=0)).to(dtype=torch.long, device=device)
        return_to_go = torch.from_numpy(np.concatenate(return_to_go, axis=0)).to(
            dtype=torch.float32, device=device
        )
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(
            dtype=torch.long, device=device
        )
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(dtype=torch.int, device=device)

        # NOTE: this is missing done_idx, as they are not used by the authors
        return states, actions, rewards, return_to_go, timesteps, mask

    # for testing
    # s, a, r, rtg, t, mask = get_batch(batch_size=8, max_len=50)
    # print(rtg.shape)
    # print(a)
    # print(r)
    # print(rtg)
    # print(t)
    # print(mask)
    # return

    model = DecisionTransformer(
        state_dim=state_dim,
        action_dim=action_dim,
        max_length=K,
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

    model.to(device)

    warmup_steps = variant["warmup_steps"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant["learning_rate"],
        weight_decay=variant["weight_decay"],
    )

    # this adjusts the learning rate as follows:
    # let's say there are 100 warmup_steps, and learning_rate is 0.5
    # then the learning rate will be:
    # 0th step: 0.05
    # 1st step: 0.06
    # 2nd step: 0.07
    # ...
    # 100th step: 0.5
    # 101st step: 0.5
    # 102nd step: 0.5
    #
    # The authors *do not* decay the learning rate in their OpenAI Gym experiments after the warmup steps!
    # They only use cosine decay for their ATARI experiments
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda steps: min((steps + 1) / warmup_steps, 1)
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        batch_size=batch_size,
        get_batch=get_batch,
        scheduler=scheduler,
        # we use cross-entropy loss as we have discrete action space
        loss_fn=lambda a_hat, a: F.cross_entropy(a_hat, a),
    )

    # actual training loop
    for iteration in range(variant["max_iters"]):
        trainer.train_iteration(num_steps=variant["num_steps_per_iter"], iter_num=iteration + 1)
        timestamp = datetime.now().replace(microsecond=0).isoformat()
        os.mkdir(f"checkpoints/{timestamp}/")
        torch.save(model.state_dict(), f"checkpoints/{timestamp}/iter_{iteration + 1:02}.pt")


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
    parser.add_argument("--max_iters", type=int, default=3)
    parser.add_argument("--num_steps_per_iter", type=int, default=10)
    parser.add_argument(
        "--scale", type=float, default=2.5
    )  # how much the rewards are scaled s.t. they fall into range [0, 10]
    parser.add_argument("--device", type=str, default="cpu")  # FIXME: "cuda" if we train on cluster

    args = parser.parse_args()
    main(variant=vars(args))
