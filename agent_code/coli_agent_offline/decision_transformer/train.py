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
def discount_cumsum(x, gamma=1.0):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum


def _get_rewards_to_go(x: np.array) -> np.array:
    """


    [0, 0, 1, 0, 5, 0] --> [6, 6, 6, 5, 5, 0]
    [1, 1, 1, 1, 1, 1] --> [6, 5, 4, 3, 2, 1]

    (try out more examples in if __name__ == __main__ to see how this function should behave)
    """
    returns_to_go = None  # TODO: this function

    return returns_to_go


def main(variant):
    device = variant.get("device", "cuda")

    max_ep_len = 401
    state_dim = 49
    action_dim = 6

    # load dataset
    # [[(s, a, r), (s, a, r)], [(s, a, r,), (s, a, r), (s, a, r)]] <-- two trajectories
    trajectories = np.load(
        "trajectories/trajectories_2022-03-23T16:38:30:372286.npy", allow_pickle=True
    )

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

            # NOTE: no vstack needed here
            # NOTE: MUST be converted to int!
            rewards.append(
                trajectory[:, 2][rng_offset : rng_offset + max_len].astype(int).reshape(1, -1, 1)
            )  # NOTE: this might have to stay a reshape and not an expand_dim

            # TODO: do we need 'terminals' or 'dones'?
            # "done signal, equal to 1 if playing the corresponding action in the state should terminate the episode"

            timesteps.append(
                np.arange(rng_offset, rng_offset + states[-1].shape[1]).reshape(
                    1, -1
                )  # [1, length_of_trajectory]
            )  # e.g. append [[399, 400, 401, 402]]
            timesteps[-1][timesteps[-1] >= max_ep_len] = (
                max_ep_len - 1
            )  # e.g. timesteps [[399, 400, 401, 402]] ==> [[399, 400, 400, 400]] when max_ep_len = 401 (1, length_of_trajectory)

            # all future rewards (even over max_len) so that the calculation of returns to go is correct
            rewards_to_go_from_offset = discount_cumsum(
                trajectory[:, 2][rng_offset:]
            )  # (length_of_trajectory from cutoff til end,)

            # append only max_len allowed sequence
            # NOTE: must be cast!
            return_to_go.append(
                rewards_to_go_from_offset[: states[-1].shape[1] + 1].astype(int).reshape(1, -1, 1)
            )  # (1, length_of_trajectory, 1)

            # there is exactly one more state seen in the current sequence than rewards in the current sequence
            # --> add one zero reward to go
            # TODO: When does this happen?
            if return_to_go[-1].shape[1] <= states[-1].shape[1]:
                return_to_go[-1] = np.concatenate(
                    [return_to_go[-1], np.zeros((1, 1, 1), dtype=int)], axis=1
                )

            sequence_len = states[-1].shape[1]

            # left-padding with zero states if our sequence is shorter than max_len
            # this happens when the offset is high and the rest of the trajectory till episode
            # finish is shorter than max_len
            # (TODO: is the padding a problem? --> 0 means free for us which would suggest that we're on a free field)
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

            # TODO: left-padding done/terminals - skip for now

            # left-padding rewards_to_go with zero reward-to-go and TODO: normalize with scale
            return_to_go[-1] = np.concatenate(
                [np.zeros((1, max_len - sequence_len, 1)), return_to_go[-1]], axis=1
            )  # / scale

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

        # print(len(states)) # batch_size because we have batchz_size sequences in the batch
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

        # TODO: add done_idx
        return states, actions, rewards, return_to_go, timesteps, mask

    get_batch(batch_size=8, max_len=50)
    return

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

    model.to(device)

    warmup_steps = variant["warmup_steps"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant["learning_rate"],
        weight_decay=variant["weight_decay"],
    )
    # TODO: understand what the scheduler does and the lambda steps:
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda steps: min((steps + 1) / warmup_steps, 1)
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        batch_size=batch_size,
        get_batch=get_batch,
        scheduler=scheduler,
        loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean(
            (a_hat - a) ** 2
        ),  # cross-entropy loss
    )

    # actual training loop
    for iteration in range(variant["max_iters"]):
        outputs = trainer.train_iteration(
            num_steps=variant["num_steps_per_iter"], iter_num=iteration + 1, print_logs=True
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
    parser.add_argument("--device", type=str, default="cpu")  # FIXME: "cuda" if we train on cluster

    args = parser.parse_args()

    # print(discount_cumsum(np.array([0, 0, 1, 0, 5, 0]), gamma=1.0))
    # print(discount_cumsum(np.array([0, 0, 1, 0, 5, 5]), gamma=1.0))
    # print(discount_cumsum(np.array([0, 0, 0, 0, 0, 0]), gamma=1.0))
    # print(discount_cumsum(np.array([1, 1, 1, 1, 1, 1]), gamma=1.0))

    main(variant=vars(args))
