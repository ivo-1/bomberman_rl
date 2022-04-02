# don't confuse with train.py files used for --train; we don't do online RL!
import argparse
import glob
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
from models.decision_transformer import DecisionTransformer
from setup_logger import dt_logger
from torch.nn import functional as F
from trainer import Trainer


def _get_rewards_to_go(rewards: np.array) -> np.array:
    """
    Returns the rewards to go from a list of rewards.

    Examples:
    [0, 0, 1, 0, 5, 0] --> [6, 6, 6, 5, 5, 0]
    [1, 1, 1, 1, 1, 1] --> [6, 5, 4, 3, 2, 1]
    """
    returns_to_go = np.zeros_like(rewards)
    returns_to_go[0] = sum(rewards)  # take sum of array as first value
    for i in range(len(returns_to_go) - 1):
        returns_to_go[i + 1] = returns_to_go[i] - rewards[i]
    return returns_to_go


def main(variant):
    """
    Main function for training the model.

    Gets the data, creates the model, trains it and saves it
    after every iteration in decision_transformer/checkpoints.
    """
    device = variant.get("device", "cuda")

    max_ep_len = 400
    state_dim = 49
    action_dim = 6

    # load dataset by combining trajectory files (every agent has its own
    # trajectory file that is generated during training)
    list_of_trajectories = glob.glob("../trajectories/*.npy")
    print(f"Found the following trajectory files: {list_of_trajectories}")
    agent_trajectories = []
    for agent_trajectory in list_of_trajectories:
        agent_trajectories.append(np.load(agent_trajectory, allow_pickle=True))

    trajectories = np.concatenate(agent_trajectories)

    # for each trajectory: save its states, length of trajectory and full return into separate lists
    states, trajectory_lens, returns = [], [], []
    for trajectory in trajectories:  # one trajectory == one episode of one agent
        states.append(trajectory[:, 0])
        trajectory_lens.append(len(trajectory[:, 0]))
        returns.append(trajectory[:, 2].sum())

    trajectory_lens, returns = np.array(trajectory_lens), np.array(returns)

    # for input normalization (later)
    states = np.vstack(np.concatenate(states, axis=0))
    state_mean, state_std = (
        np.mean(states, axis=0),
        np.std(states, axis=0) + 1e-6,
    )  # adding small number in case std is 0 (to prevent ZeroDivisionError)

    # save for later access in callbacks.py where we need to normalize states again
    np.save("../data/coli_states_mean.npy", state_mean)
    np.save("../data/coli_states_std.npy", state_std)

    num_timesteps = sum(trajectory_lens)
    num_trajectories = len(
        trajectory_lens
    )  # we don't do any top-k % like the paper does, so just take all

    print("=" * 50)
    print(f"{len(trajectory_lens)} trajectories, {num_timesteps} timesteps found")
    print(f"Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}")
    print(f"Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}")
    print("=" * 50)

    K = variant["K"]
    batch_size = variant["batch_size"]  # NOTE: needed for Trainer

    # probability of sampling trajectory weighted by the number of timesteps in that trajectory
    p_sample = trajectory_lens / num_timesteps

    def get_batch(batch_size=batch_size, max_len=K):
        """
        Returns a batch of sub-trajectories, meaning
        random trajectories of length max_len from the training data.

        More precisely:

        states: [batch_size, max_len, state_dim]
        actions: [batch_size, max_len, action_dim]
        return_to_go: [batch_size, max_len + 1, 1]
        timesteps: [batch_size, max_len]
        mask: [batch_size, max_len]
        """
        # get batch_size random trajectory indices
        trajectory_idx = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,
        )

        states, actions, return_to_go, timesteps, mask = (
            [],
            [],
            [],
            [],
            [],
        )

        # construct a batch by iterating over the randomly chosen trajectories
        for trajectory_index in trajectory_idx:
            trajectory = trajectories[
                trajectory_index
            ]  # current trajectory: [length_of_trajectory, 3] where 3: (state, action, return)

            # get random int in [0, traj_len- 1]
            rng_offset = random.randint(0, trajectory[:, 0].shape[0] - 1)

            # get the states and actions from the sequence of randomly chosen sequence starting at rng_offset
            states.append(
                np.vstack(trajectory[:, 0][rng_offset : rng_offset + max_len]).reshape(
                    1, -1, state_dim
                )  # (1, length_of_sequence, state_dim)
            )
            actions.append(
                np.vstack(trajectory[:, 1][rng_offset : rng_offset + max_len]).reshape(
                    1, -1, action_dim
                )
            )

            sequence_len = states[-1].shape[
                1
            ]  # actual length of current sequence (might be shorter than max_len)

            timesteps.append(
                np.arange(rng_offset, rng_offset + sequence_len).reshape(1, -1)  # [1, sequence_len]
            )

            timesteps[-1][timesteps[-1] >= max_ep_len] = (
                max_ep_len - 1
            )  # e.g. timesteps[-1] was [[396, 397, 398, 399, 400]] ==> [[396, 397, 398, 399, 399]] when max_ep_len = 400 (1, length_of_trajectory)

            # all future rewards (will usually be bigger than max_len) so that the calculation of returns to go is correct
            rewards_to_go_from_offset = _get_rewards_to_go(trajectory[:, 2][rng_offset:])

            # append only max_len allowed sequence of rewards to go
            # NOTE: must be cast to int!
            return_to_go.append(
                rewards_to_go_from_offset[: sequence_len + 1].astype(int).reshape(1, -1, 1)
            )  # (1, sequence_len + 1, 1) unless sequence_len <= max_len

            # this happens when the current sampled sequence is
            # shorter than max_len, i.e. because the offset was high
            # enough to reach over or exactly until the end of the trajectory
            # ==> pad it, s.t. rtg[-1] has shape (1, sequence_len + 1, 1)
            if return_to_go[-1].shape[1] == sequence_len:
                return_to_go[-1] = np.concatenate(
                    [return_to_go[-1], np.zeros((1, 1, 1), dtype=int)], axis=1
                )

            # left-padding with zero states if sequence_len is shorter than max_len
            states[-1] = np.concatenate(
                [np.zeros((1, max_len - sequence_len, state_dim), dtype=int), states[-1]], axis=1
            )
            # normalization of states
            states[-1] = (states[-1] - state_mean) / state_std

            # left-padding with dummy action (not one-hot encoded, but just whole vector -10)
            actions[-1] = np.concatenate(
                [np.ones((1, max_len - sequence_len, action_dim)) * -10, actions[-1]], axis=1
            )

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

        # concatenate all the stuff of all the sequences in the batch to be behind each other and prepare them to be sent to torch
        states = torch.from_numpy(np.concatenate(states, axis=0)).to(
            dtype=torch.float32, device=device
        )

        actions = torch.from_numpy(np.concatenate(actions, axis=0)).to(
            dtype=torch.float32, device=device
        )

        return_to_go = torch.from_numpy(np.concatenate(return_to_go, axis=0)).to(
            dtype=torch.float32, device=device
        )
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(
            dtype=torch.long, device=device
        )
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(dtype=torch.int, device=device)

        return states, actions, return_to_go, timesteps, mask

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
        embd_pdrop=variant["dropout"],
    )

    # NOTE: uncomment if you want to continue training from a checkpoint
    # path = "<PATH_TO_CHECKPOINT>"
    # model.load_state_dict(torch.load(path))

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

    timestamp = time.time()
    isoformat_time = datetime.fromtimestamp(timestamp).replace(microsecond=0).isoformat()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        batch_size=batch_size,
        get_batch=get_batch,
        scheduler=scheduler,
        # we use cross-entropy loss as we have discrete action space
        loss_fn=lambda a_hat, a: F.cross_entropy(a_hat, a),
        start_time=timestamp,
    )

    try:
        os.mkdir(f"checkpoints/{isoformat_time}/")
        os.mkdir(f"plots/{isoformat_time}/")
    except FileExistsError:
        dt_logger.warning("Tried to create already existing checkpoint or plots folder.")

    # actual training loop
    for iteration in range(variant["max_iters"]):
        trainer.train_iteration(num_steps=variant["num_steps_per_iter"], iter_num=iteration)
        torch.save(model.state_dict(), f"checkpoints/{isoformat_time}/iter_{iteration + 1:02}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # training hyperparameters
    parser.add_argument("--K", type=int, default=20)  # the size of the context window
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--embed_dim", type=int, default=128
    )  # size of timestep, return, state, action embeddings
    parser.add_argument("--n_layer", type=int, default=3)  # GPT2
    parser.add_argument("--n_head", type=int, default=1)  # GPT2
    parser.add_argument("--activation_function", type=str, default="relu")
    parser.add_argument(
        "--dropout", type=float, default=0.1
    )  # probability of dropping out residuals, attentions and embeddings in GPT2
    parser.add_argument(
        "--learning_rate", "-lr", type=float, default=1e-4
    )  # learning rate after warmup steps (no decay after warmup)
    parser.add_argument("--weight_decay", "-wd", type=float, default=1e-4)  # for AdamW optimizer
    parser.add_argument(
        "--warmup_steps", type=int, default=10000
    )  # warming up the learning rate for warmup_steps
    parser.add_argument(
        "--max_iters", type=int, default=20
    )  # number of iterations (kind of like "epochs" although we don't see all the dataset per epoch)
    parser.add_argument(
        "--num_steps_per_iter", type=int, default=10000
    )  # how many batches of size batch_size should be sampled per iteration
    parser.add_argument(
        "--scale", type=float, default=2.5
    )  # how much the rewards are scaled s.t. they fall into range [0, 10]
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()
    main(variant=vars(args))
