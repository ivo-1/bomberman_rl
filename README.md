# Reinforcement Learning with Old and New Methods 

This project implements two Reinforcement Learning approaches, Q-Learning and a Decision Transformer, to a simplified version of the classic arcade game Bomberman.
The implementation was done as part of the 2021/2022 lecture Fundamentals of Machine Learning held by Prof. Dr. Ullrich Köthe at Heidelberg University.

## Installation

After cloning the repository, we recommend using [poetry](https://python-poetry.org/) to install the necessary dependencies. Poetry itself can be installed on unix systems with the following command:

```shell
curl -sSL https://install.python-poetry.org | python3
```

Then install the packages by navigating to the `bomberman_rl` home directory and running `poetry install`.

You can otherwise try installing the packages using the provided `requirements.txt`, however, this is not well-tested and may lead to version issues.

No other preparation is necessary.

## Usage

You can use the code in two ways: Playing test runs with our trained models and q-tables or training new ones yourself. Our Q-Learning agent (agent submitted to the tournament) is called `coli_agent` and our Decision Transformer agent is called `coli_agent_offline`.

### Training

Go to the `bomberman_rl` home directory.
If you decided to use poetry, now first run `poetry shell` once, which will active the environment. (You can quit the environment again the same way you would exit a shell.)

#### Q-Learning Agent
If you want to train the Q-Learning agent, stay in `bomberman_rl/` and run any training version you want like so:

```shell
python main.py play --train 1 --my-agent coli_agent --n-rounds 251
```

Depending on the number of rounds specified, this may take a while to start because our setup is somewhat computationally expensive.

Consult `python main.py play --h` for information about available flags.

There is also a _new flag_ available that we introduced. It's called `--continue_training`. If you pass it to your training command, the latest q-table in the directory `agent_code/coli_agent/q_tables` will be used and further updated, i.e. continued to be trained. Otherwise, a new q-table will be created in that directory. After every 250 episodes, the current q-table status is automatically saved. The same goes for plots, which are automatically created during training and saved in the directory `agent_code/coli_agent/plots`. For this reason, we recommend _training with multiples of 250 episodes_.

**Debugging/Analysis**

You may want to change the logging level. You can do so in `settings.py`. Be aware that the DEBUG level will provide you with quite a lot of messages both during training and during testing. If you plan to play a large number of episodes (upwards of 10,000-20,000, depending on your machine) this may lead to memory issues. We recommend lowering the logging level in such a case, or using our `--continue_training` flag to split training into multiple parts.

#### Decision Transformer
If you want to train your own decision transformer, navigate to `agent_code/coli_agent_offline/decision_transformer/`, then run training like this:

```shell
python train.py --warmup_steps 10000 --max_iters 20 --num_steps_per_iter 10000 [--device cuda]
```

You should see a printed message about trajectories being loaded.

Refer to `python train.py -h` for a list of available flags.

There is currently no dynamic option available to continue training with the last checkpoint, you would have manually set this in `coli_agent_offline/decision_transformer/train.py`. However, checkpoints _are_ automatically saved after every iteration, as well as the correponding plots. In our setting, iterations serve no other purpose than this.

### Inference

Both agents can be used the same way in the test case. Navigate to the `bomberman_rl` home directory, (don't forget to have your environment active,) then run:

```shell
python main.py play --my-agent <coli_agent/coli_agent_offline>
```

plus any additional flags you want, such as `--turn-based`.

## Alternative Q-Learning agent versions

We attempted to design our Q-Learning agent a total of three times. On our `master` branch you find our best-performing version that we recommend using. If you, however, for some reason want to experiment with our other attempts, you can find these on the branches `feature_set_1` and `feature_set_3` and use them the same as explained above.

## Credit
Our Decision Transformer agent is based on the original Decision Transformer code by Chen et al. Find their GitHub repository [here](https://github.com/kzl/decision-transformer) and the paper it belongs to [here](https://sites.google.com/berkeley.edu/decision-transformer). We have adapted their code to our task and simplified elements where we didn't need all functionalities, but the GPT2 model basis we use is the same as theirs (except for how tokenization is treated). See our project report for further explanation of this.

## Authors
Aileen Reichelt \
Ivo Schaper \
Kieu Trang Nguyen Vu

**Contact:** {reichelt, schaper, vu}@cl.uni-heidelberg.de
