import time

import numpy as np
import torch
from setup_logger import dt_logger


class Trainer:
    """pseudo docstring"""

    def __init__(self, model, optimizer, batch_size, get_batch, loss_fn, scheduler, eval_fns=None):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.start_time = time.time()

    def train_iteration(self, num_steps, iter_num=0):

        dt_logger.info(f"====================== Iteration {iter_num} ==========================")

        train_losses = []

        train_start = time.time()

        self.model.train()
        for _ in range(num_steps):
            train_losses.append(self.train_step())
            self.scheduler.step()

        dt_logger.info(f"training time: {time.time() - train_start}")

        eval_start = time.time()

        self.model.eval()
        for eval_fn in self.eval_fns:
            outputs = eval_fn(self.model)
            for k, v in outputs.items():
                dt_logger.info(f"evaluation of {k}: {v}")

        dt_logger.info(f"total time: {time.time() - self.start_time}")
        dt_logger.info(f"evaluation time: {time.time() - eval_start}")
        dt_logger.info(f"training loss mean: {np.mean(train_losses)}")
        dt_logger.info(f"training loss std: {np.std(train_losses)}")

    def train_step(self):
        # get data for this training iteration
        states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(
            self.batch_size
        )
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

        # action_preds is originally (?, ?, number of possible actions)
        act_dim = action_preds.shape[2]  # here: 6
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        # is defined in train.py as cross-entropy loss
        loss = self.loss_fn(action_preds, action_target)

        # actual training:
        # resets gradients for this training step,
        # computes the new gradient and its norm
        # and performs parameter update
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        self.optimizer.step()

        with torch.no_grad():  # TODO change to cross entropy loss
            dt_logger.info(
                f"training action error: {torch.mean((action_preds-action_target)**2).detach().cpu().item()}"
            )

        return loss.detach().cpu().item()