from tkinter.messagebox import NO

import torch
import torch.nn as nn
import transformers

from .trajectory_gpt2 import GPT2Model


class DecisionTransformer(nn.Module):
    """
    PyTorch Model which uses GPT2 to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
        self,
        state_dim,  # how many entries the feature vector has - 49
        action_dim,  # how many actions one can take - 6 NOTE: actions are one-hot-encoded
        hidden_size,  # size of timestep, return, state, action embeddings
        max_length=None,  # size of context window
        max_ep_len=400,  # game of bomberman lasts max. 400 steps
        # squish values of action vector output to be between -1 and 1
        # NOTE: these are the "logits", as the cross-entropy loss includes
        # a softmax layer it is correct to not use softmax before
        action_tanh=True,
        **kwargs,
    ):
        super().__init__()  # inherit usual stuff from nn.Module

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_length = max_length

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter because the vocab is not used
            n_embd=hidden_size,  # tell GPT2 dimension of our embeddings (of timestep, return, state, action)
            **kwargs,
        )

        # NOTE: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since these are simply added here in line 43)
        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)  # own positional embedding
        self.embed_return = nn.Linear(1, hidden_size)  # return is just a number, hence dimension 1
        self.embed_state = nn.Linear(self.state_dim, hidden_size)
        self.embed_action = nn.Linear(self.action_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)  # Layer Normalization

        # NOTE: we don't care about predicting states and returns, only about predicting actions
        self.tanh = (
            [nn.Tanh()] if action_tanh else []
        )  # needs to be lists, so we can unpack them (else case cannot just be None)
        self.predict_action = nn.Sequential(
            nn.Linear(hidden_size, self.action_dim), *self.tanh
        )  # nn.Linear --> turn hidden vector into action vector of size action_dim and then run tanh over the result if action_tanh is True

    def forward(self, states, actions, returns_to_go, timesteps, attention_mask=None):
        batch_size, seq_length = (
            states.shape[0],
            states.shape[1],
        )  # NOTE: states are (B, K, state_dim) where B = batch_size, K=context size, state_dim=49

        # NOTE: attention mask for GPT: 1 if can be attended to, 0 if not - not to confuse with
        # masking *within* the transformer to calculate attention scores

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings, i.e. they are simply added
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        stacked_inputs = (
            torch.stack(
                (returns_embeddings, state_embeddings, action_embeddings), dim=1
            )  # (B, 3, K, hidden_size) where 3 is for return, state, action
            .permute(0, 2, 1, 3)  # (B, K, 3, hidden_size)
            .reshape(batch_size, -1, self.hidden_size)  # (B, 3*K, hidden_size)
        )
        stacked_inputs = self.embed_ln(stacked_inputs)  # layer normalization

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = (
            torch.stack((attention_mask, attention_mask, attention_mask), dim=1)
            .permute(0, 2, 1)
            .reshape(batch_size, -1)
        )  # (B, 3*K)
        # NOTE: input embeddings to the transformer are just
        # stacked embeddings of (R, s, a) and all of that embedded linearly via layer normalization
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs["last_hidden_state"]  # (B, 3*K, hidden_size)

        # reshape x so that the second dimension corresponds to the original stacked inputs
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(
            0, 2, 1, 3
        )  # (B, 3, K, hidden_size)

        # [:, 1] contains the state embeddings which were created by the transformer which
        # uses self-attention and thus contains informataion of *all* previous states,
        # actions and rewards
        action_preds = self.predict_action(
            x[:, 1]
        )  # predict next action given state x[:, 1] which has shape (B, K, hidden_size)
        return action_preds  # (B, K, action_dim)

    # only at INFERENCE, where we start without a sequence of length K
    # and where we will have sequences longer than K
    # i.e. this function makes sure everything is padded to length K
    # just like the model was trained only with sequences of length K
    def get_action(self, states, actions, returns_to_go, timesteps):
        states = states.reshape(1, -1, self.state_dim)  # (1, sequence_len, state_dim)
        actions = actions.reshape(
            1, -1, self.action_dim
        )  # (1, sequence_len - 1, action_dim) because of course the action hasn't been taken yet
        returns_to_go = returns_to_go.reshape(1, -1, 1)  # (1, sequence_len, 1)
        timesteps = timesteps.reshape(1, -1)  # (1, sequence_len - 1)

        # only consider the last max_length tokens
        states = states[:, -self.max_length :]
        actions = actions[:, -self.max_length :]
        returns_to_go = returns_to_go[:, -self.max_length :]
        timesteps = timesteps[:, -self.max_length :]

        # left-pad everything to max_length if we haven't yet seen max_length tokens and
        # conceal the padding with the attention mask
        attention_mask = torch.cat(
            [torch.zeros(self.max_length - states.shape[1]), torch.ones(states.shape[1])]
        )
        attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(
            1, -1
        )  # (1, max_length)
        states = torch.cat(
            [
                torch.zeros(
                    (states.shape[0], self.max_length - states.shape[1], self.state_dim),
                    device=states.device,
                ),
                states,
            ],
            dim=1,
        ).to(dtype=torch.float32)
        actions = torch.cat(
            [
                torch.zeros(
                    (actions.shape[0], self.max_length - actions.shape[1], self.action_dim),
                    device=actions.device,
                ),
                actions,
            ],
            dim=1,
        ).to(dtype=torch.float32)
        returns_to_go = torch.cat(
            [
                torch.zeros(
                    (returns_to_go.shape[0], self.max_length - returns_to_go.shape[1], 1),
                    device=returns_to_go.device,
                ),
                returns_to_go,
            ],
            dim=1,
        ).to(dtype=torch.float32)
        timesteps = torch.cat(
            [
                torch.zeros(
                    (timesteps.shape[0], self.max_length - timesteps.shape[1]),
                    device=timesteps.device,
                ),
                timesteps,
            ],
            dim=1,
        ).to(dtype=torch.long)

        # now that we have padded everything to the same context window the model
        # was trained with, we can call the model
        action_preds = self.forward(
            states, actions, returns_to_go, timesteps, attention_mask=attention_mask
        )
        return action_preds[
            0, -1
        ]  # -1 because we want the prediction for the action that hasn't happened yet (the last one)
