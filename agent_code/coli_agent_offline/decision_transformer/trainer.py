import time
from datetime import datetime

import numpy as np
import torch
from plots import plot_loss
from setup_logger import dt_logger


class Trainer:
    """Trainer class to capsulate training definition from main file."""

    def __init__(
        self, model, optimizer, batch_size, get_batch, loss_fn, scheduler, start_time, eval_fns=None
    ):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.start_time = start_time
        self.train_losses_all_iterations = []

    def train_iteration(self, num_steps, iter_num=0):
        """This method calls the train_step() method for num_steps."""

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

        eval_start = time.time()

        self.model.eval()
        for eval_fn in self.eval_fns:
            outputs = eval_fn(self.model)
            for k, v in outputs.items():
                dt_logger.info(f"evaluation of {k}: {v}")

        # times may not be 100 % accurate due to where they are initialized
        dt_logger.info(f"total time: {round(time.time() - self.start_time, 2)} s")
        dt_logger.info(f"evaluation time: {round(time.time() - eval_start, 2)} s")
        dt_logger.info(f"training loss mean: {round(np.mean(train_losses), 4)}")
        dt_logger.info(f"training loss std: {round(np.std(train_losses), 4)}")

    def train_step(self):
        """In this method the loss is getting computed for each step."""

        # get data for this training iteration
        states, actions, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
        # in order to calculate loss, we need targets and predictions
        # cloning to circumvent pass-by-reference
        action_target = actions.clone()

        # feed data into model
        action_preds = self.model.forward(
            states,
            actions,
            rtg[:, :-1],
            timesteps,
            attention_mask=attention_mask,
        )  # rtg[:,:-1] = last return-to-go in the context window of each trajectory
        # the authors also get state and reward predictions but we don't need those

        # action_preds is originally (trajectory, state, actions)
        # gets reshaped to (trajectory * state, actions), i.e. flattened by one dimension
        # attention_mask is reshaped to the same dimensionality and used to pick those actions
        # which have an attention greater than 0, i.e. are supposed to be remembered
        act_dim = action_preds.shape[2]  # here: 6
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]  # (N, 6)
        action_target = torch.argmax(action_target, axis=1)  # (N, )

        # is defined in train.py as cross-entropy loss
        loss = self.loss_fn(action_preds, action_target)

        detached_loss = loss.detach().cpu().item()

        # actual training:
        # resets gradients for this training step,
        # computes the new gradient and its norm
        # and performs parameter update
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        self.optimizer.step()

        with torch.no_grad():
            dt_logger.info(f"training action error: {round(detached_loss, 4)}")

        return detached_loss
