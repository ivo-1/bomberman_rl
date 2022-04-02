import time
from datetime import datetime

import numpy as np
import torch
from plots import plot_loss
from setup_logger import dt_logger


class Trainer:
    """Trainer class to capsulate training definition from main file."""

    def __init__(self, model, optimizer, batch_size, get_batch, loss_fn, scheduler, start_time):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.start_time = start_time
        self.train_losses_all_iterations = []

    def train_iteration(self, num_steps: int, iter_num=0) -> None:
        """Calls the train_step() method for num_steps and performs logging and plotting."""

        dt_logger.info(f"====================== Iteration {iter_num} ==========================")

        train_losses = []
        train_start = time.time()

        self.model.train()
        for _ in range(num_steps):
            train_losses.append(self.train_step())
            self.scheduler.step()

        dt_logger.info(f"training time: {time.time() - train_start}")

        isoformat_time = datetime.fromtimestamp(self.start_time).replace(microsecond=0).isoformat()
        self.train_losses_all_iterations.extend(train_losses)  # for plotting

        plot_loss(train_losses, iso_time=isoformat_time, iteration=iter_num + 1, version="current")
        plot_loss(train_losses, iso_time=isoformat_time, iteration=iter_num + 1, version="detail")
        plot_loss(
            self.train_losses_all_iterations,
            iso_time=isoformat_time,
            iteration=iter_num + 1,
            version="so far",
        )

        # times may not be 100 % accurate due to where they are initialized
        dt_logger.info(f"total time: {round(time.time() - self.start_time, 2)} s")
        dt_logger.info(f"training loss mean: {round(np.mean(train_losses), 4)}")
        dt_logger.info(f"training loss std: {round(np.std(train_losses), 4)}")

    def train_step(self) -> float:
        """
        Computes the loss for a single training step and performs a single optimization
        step via backpropagation and gradient descent.

        Returns the loss for each step.
        """

        # get data for this training iteration
        states, actions, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)

        # cloning to circumvent pass-by-reference
        action_target = actions.clone()

        # feed data into model
        action_preds = self.model.forward(
            states,
            actions,
            rtg[
                :, :-1
            ],  # exclude last return-to-go of each trajectory (so that shape is (batch_size, K, 1))
            timesteps,
            attention_mask=attention_mask,
        )  # these are logits (not probabilities)

        action_dim = action_preds.shape[2]

        # action_preds is originally (batch_size, K, action_dim)
        # this is first reshaped to (batch_size * K, actions), i.e. we
        # forget the batches and just combine them for the loss computation,
        # and then remove all action predictions which were masked out (i.e. 0)
        action_preds = action_preds.reshape(-1, action_dim)[attention_mask.reshape(-1) > 0]

        # action_target is handled the same way but finally we only take the
        # argmax (the index were the one-hot-encoded vector is 1), because the loss
        # function demands that we just compare to the correct class index
        action_target = action_target.reshape(-1, action_dim)[
            attention_mask.reshape(-1) > 0
        ]  # (batch_size * K, 6)
        action_target = torch.argmax(action_target, axis=1)  # (batch_size * K, )

        # is defined in train.py as cross-entropy loss
        loss = self.loss_fn(action_preds, action_target)

        # actual training:
        # resets gradients for this training step,
        # computes the new gradient and its norm
        # and performs parameter update via backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        self.optimizer.step()

        detached_loss = loss.detach().cpu().item()
        with torch.no_grad():
            dt_logger.info(f"training action error: {round(detached_loss, 4)}")

        return detached_loss
