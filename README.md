## Coin Flip Networks

This is the code for the paper **Flipping Coins to Estimate Pseudocounts for Exploration in Reinforcement Learning**, by Sam Lobel, Akhil Bagaria, and George Konidaris, to be presented at ICML 2023. This repository includes all the code needed to run all experiments described in our paper, as well as the code to make the main paper plots.

### Installation
```
sudo apt-get install libxml2-dev libxslt-dev # sometimes required for dm-control
python3 -m venv venv
source ./venv/bin/activate
pip install --upgrade pip
pip install --upgrade jax==0.3.15 jaxlib==0.3.15 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -r requirements.txt
pip install -e .
pip install "gym[atari,mujoco]==0.23.1" # (separte because of mujoco install weirdness. Also, prepend "noglob" for zsh)
pip install --no-deps --ignore-requires-python git+https://github.com/camall3n/visgrid@6f7f3a6373e478dbc64e27a692d75f499e5870e0
```

You can add cuda to jax by following the instructions on the Jax Github page [here](https://github.com/google/jax#instructions)

### Running Experiments
Sample command for running CFN on Visual GridWorld:

```
python bonus_based_exploration/train.py --gin_files configs/gridworld_configs/rainbow_coinflip.gin --base_dir dopamine_logs/gridworld/rainbow_coinflip/42/full_runs --gin_bindings 'create_exploration_agent.debug_mode = True' --gin_bindings 'GridWorldEnv.size = 42' --sub_dir gridworld_coinflip_full_01__GridWorldEnv.size_42
```

Sample command for running CFN on the Ant Umaze:

```
XLA_PYTHON_CLIENT_PREALLOCATE=false python bonus_based_exploration/cc_intrinsic_motivation/train.py --gin_files configs/continuous_control/ant/sac_coinflip.gin --base_dir dopamine_logs/d4rl/ant/sac_coinflip/full_runs --gin_bindings 'ContinuousRunner.num_iterations = 400' --sub_dir ant_coinflip_full_1__ContinuousRunner.numiterations_400
```

The core implementation of the Coin Flip Network bonus module can be found in `intrinsic_motivation/intrinsic_rewards.py`. We use a fork of the [onager](https://github.com/camall3n/onager) package to orchestrate runs locally or on a slurm/gridengine cluster. All run commands can be generated from the `onager prelaunch` commands defined in `prelaunch_commands.txt`, and then run with `onager launch ...` or by simply running the printed commands. Note that with default configs, the monte experiments take a lot of memory, because of the CFN replay buffer! You can overwrite this with `--gin_bindings 'CoinFlipCounterIntrinsicReward.intrinsic_replay_buffer_size = 1000000'` for example.


### Making Plots
Logs are stored in folders defined by the `--base_dir` and `--sub_dir` flags. All plotting code can be found in the `scripts` directory. To make exploratory plots, look in the `plotting_utils.py` and `comparison_plotting.py` files. To make plots from the paper, use `make_icml_2023_paper_plots.py`, modifying directories to match where your results are stored.



### Acknowledgements

This repository builds off of the [code](https://github.com/google-research/google-research/tree/master/bonus_based_exploration) for the paper [On Bonus Based Exploration Methods In The Arcade Learning Environment](https://openreview.net/forum?id=BJewlyStDr), and additionally copies environment-building code from the [visgrid](https://github.com/camall3n/visgrid).


### Citation

You may cite us at

```
@inproceedings{Lobel2023Flipping,
title={Flipping Coins to Estimate Pseudocounts for Exploration in Reinforcement Learning},
author={Sam Lobel and Akhil Bagaria and George Konidaris},
booktitle={International Conference on Machine Learning},
year={2023},
}
```
(URL to come!)

