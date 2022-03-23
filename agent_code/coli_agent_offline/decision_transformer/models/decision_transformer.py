import torch
import torch.nn as nn
import transformers
from models.trajectory_gpt2 import GPT2Model


class DecisionTransformer(nn.Module):
    """
    PyTorch Model

    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
        self,
        state_dim,  # how many entries the feature vector has - 49
        action_dim,  # how many actions one can take - 6 NOTE: actions are one-hot-encoded
        hidden_size,  # size of positional embeddings
        max_length=None,  # context window
        max_ep_len=401,  # game of bomberman lasts max. 401 steps
        action_tanh=True,  # squish values of action vector output to be between -1 and 1 TODO: this must later be run through softmax somewhere@
        **kwargs,
    ):
        super().__init__()  # inherit usual stuff from nn.Module

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_length = max_length

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # TODO: explain this:  doesn't matter -- we don't use the vocab
            n_embd=hidden_size,  # tell GPT2 dimension of embeddings (of return, state, action but not of timestep [positional] embedding)
            **kwargs,
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)  # own positional embedding
        self.embed_return = nn.Linear(1, hidden_size)  # return is just a number --> 1
        self.embed_state = nn.Linear(self.state_dim, hidden_size)
        self.embed_action = nn.Linear(self.action_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)  # Layer Normalization

        # note: we don't predict states or returns for the paper
        # NOTE: we don't care about predicting states and returns, only actions
        # self.predict_state = nn.Linear(hidden_size, self.state_dim)
        self.tanh = (
            [nn.Tanh()] if action_tanh else []
        )  # needs to be lists, so we can unpack them (else case cannot be None)
        self.predict_action = nn.Sequential(
            nn.Linear(hidden_size, self.action_dim), *self.tanh
        )  # nn.Linear --> turn hidden vector into one-hot encoded action vector - run tanh over the result if action_tanh is True
        # self.predict_return = nn.Linear(hidden_size, 1)

    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None):

        batch_size, seq_length = (
            states.shape[0],
            states.shape[1],
        )  # NOTE: states are (B, K, state_dim) where B = batch_size, K=context size, state_dim=49

        if (
            attention_mask is None
        ):  # NOTE: if for some reason get_batch doesn't return an attention_mask
            # attention mask for GPT: 1 if can be attended to, 0 if not - not to confuse with
            # masking *within* the transformer to calculate attention
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions

        print(
            f"Stack shape: {torch.stack((returns_embeddings, state_embeddings, action_embeddings), dim=1).shape}"
        )
        print(
            f"Permuted stack shape: {torch.stack((returns_embeddings, state_embeddings, action_embeddings), dim=1).permute(0, 2, 1, 3).shape}"
        )
        stacked_inputs = (
            torch.stack((returns_embeddings, state_embeddings, action_embeddings), dim=1)  # ()
            .permute(0, 2, 1, 3)
            .reshape(
                batch_size, 3 * seq_length, self.hidden_size
            )  # NOTE: 3*seq_length because one step consists of 3 tokens returns_to_go, action and state
        )
        print(f"Stacked+permuted+reshaped shape: {stacked_inputs.shape}")
        stacked_inputs = self.embed_ln(stacked_inputs)  # layer normalization

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = (
            torch.stack((attention_mask, attention_mask, attention_mask), dim=1)
            .permute(0, 2, 1)
            .reshape(batch_size, 3 * seq_length)
        )

        # NOTE: input embeddings = stacked embeddings of (R, s, a) and all of that embedded linearly
        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs["last_hidden_state"]

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        # return_preds = self.predict_return(x[:,2])  # predict next return given state and action
        # state_preds = self.predict_state(x[:,2])    # predict next state given state and action
        action_preds = self.predict_action(x[:, 1])  # predict next action given state

        return action_preds  # state_preds, action_preds, return_preds

    def get_action(
        self, states, actions, rewards, returns_to_go, timesteps, **kwargs
    ):  # only at INFERENCE
        # we don't care about the past rewards in this model

        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.action_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:  # we have given a context window
            states = states[:, -self.max_length :]
            actions = actions[:, -self.max_length :]
            returns_to_go = returns_to_go[:, -self.max_length :]
            timesteps = timesteps[:, -self.max_length :]

            # pad all tokens to sequence length
            attention_mask = torch.cat(
                [torch.zeros(self.max_length - states.shape[1]), torch.ones(states.shape[1])]
            )
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(
                1, -1
            )
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
        else:  # there is no context window
            attention_mask = None

        _, action_preds, return_preds = self.forward(
            states, actions, None, returns_to_go, timesteps, attention_mask=attention_mask, **kwargs
        )

        return action_preds[0, -1]
