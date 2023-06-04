#!/bin/bash

# This file contains all the `onager prelaunch` commands needed to generate the data in our paper.
# This file can either be run as is, or lines can be ran individually

# gridworld
onager prelaunch +jobname gridworld_coinflip_full +command "PYTHONPATH=.. python -m bonus_based_exploration.train --gin_files configs/gridworld_configs/rainbow_coinflip.gin --base_dir dopamine_logs/gridworld/rainbow_coinflip/42/full_runs --gin_bindings 'create_exploration_agent.debug_mode = True'" +gin-arg GridWorldEnv.size 42 42 42 42 42 42 42 42 42 42 +tag --sub_dir
onager prelaunch +jobname gridworld_rnd_full +command "PYTHONPATH=.. python -m bonus_based_exploration.train --gin_files configs/gridworld_configs/rainbow_rnd.gin --base_dir dopamine_logs/gridworld/rainbow_rnd/42/full_runs --gin_bindings 'create_exploration_agent.debug_mode = True'" +gin-arg GridWorldEnv.size 42 42 42 42 42 42 42 42 42 42 +tag --sub_dir
onager prelaunch +jobname gridworld_pixelcnn_full +command "PYTHONPATH=.. python -m bonus_based_exploration.train --gin_files configs/gridworld_configs/rainbow_pixelcnn.gin --base_dir dopamine_logs/gridworld/rainbow_pixelcnn/42/full_runs --gin_bindings 'create_exploration_agent.debug_mode = True'" +gin-arg GridWorldEnv.size 42 42 42 42 42 42 42 42 42 42 +tag --sub_dir
onager prelaunch +jobname gridworld_vanilla_full +command "PYTHONPATH=.. python -m bonus_based_exploration.train --gin_files configs/gridworld_configs/rainbow_vanilla.gin --base_dir dopamine_logs/gridworld/rainbow_vanilla/42/full_runs --gin_bindings 'create_exploration_agent.debug_mode = True'" +gin-arg GridWorldEnv.size 42 42 42 42 42 42 42 42 42 42 +tag --sub_dir

# Monte
onager prelaunch +jobname monte_coinflip_full +command "PYTHONPATH=.. python -m bonus_based_exploration.train --gin_files configs/monte_configs/rainbow_coinflip.gin --base_dir dopamine_logs/monte/rainbow_coinflip/full_runs --gin_bindings 'create_exploration_agent.debug_mode = True'" +gin-arg Runner.num_iterations 200 200 200 200 200 200 200 200 200 200 200 200  +tag --sub_dir
onager prelaunch +jobname monte_rnd_full +command "PYTHONPATH=.. python -m bonus_based_exploration.train --gin_files configs/monte_configs/rainbow_rnd.gin --base_dir dopamine_logs/monte/rainbow_rnd/full_runs --gin_bindings 'create_exploration_agent.debug_mode = True'" +gin-arg Runner.num_iterations 200 200 200 200 200 200 200 200 200 200 200 200  +tag --sub_dir
onager prelaunch +jobname monte_pixelcnn_full +command "PYTHONPATH=.. python -m bonus_based_exploration.train --gin_files configs/monte_configs/rainbow_pixelcnn.gin --base_dir dopamine_logs/monte/rainbow_pixelcnn/full_runs --gin_bindings 'create_exploration_agent.debug_mode = True'" +gin-arg Runner.num_iterations 200 200 200 200 200 200 200 200 200 200 200 200  +tag --sub_dir
onager prelaunch +jobname monte_vanilla_full +command "PYTHONPATH=.. python -m bonus_based_exploration.train --gin_files configs/monte_configs/rainbow_vanilla.gin --base_dir dopamine_logs/monte/rainbow_vanilla/full_runs --gin_bindings 'create_exploration_agent.debug_mode = True'" +gin-arg Runner.num_iterations 200 200 200 200 200 200 200 200 200 200 200 200  +tag --sub_dir

# Fetch (just coinflip and push, extrapolate for others)
onager prelaunch +jobname fetch_push_coinflip_full_mode_-1 +command "XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=.. python -m bonus_based_exploration.cc_intrinsic_motivation.train --gin_files configs/continuous_control/fetch/sac_coinflip.gin --base_dir dopamine_logs/fetch/push/sac_coinflip/mode_-1/full_runs" +gin-arg FetchEnvWrapper.mode -1 -1 -1 -1 -1 -1 -1 -1 -1 +tag --sub_dir
onager prelaunch +jobname fetch_push_coinflip_full_mode_1 +command "XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=.. python -m bonus_based_exploration.cc_intrinsic_motivation.train --gin_files configs/continuous_control/fetch/sac_coinflip.gin --base_dir dopamine_logs/fetch/push/sac_coinflip/mode_1/full_runs" +gin-arg FetchEnvWrapper.mode 1 1 1 1 1 1 1 1 1 +tag --sub_dir
onager prelaunch +jobname fetch_push_coinflip_full_mode_2 +command "XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=.. python -m bonus_based_exploration.cc_intrinsic_motivation.train --gin_files configs/continuous_control/fetch/sac_coinflip.gin --base_dir dopamine_logs/fetch/push/sac_coinflip/mode_2/full_runs" +gin-arg FetchEnvWrapper.mode 2 2 2 2 2 2 2 2 2 +tag --sub_dir

# D4RL tasks (just coinflip and ant, extrapolate for others)
onager prelaunch +jobname ant_coinflip_full +command "XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=.. python -m bonus_based_exploration.cc_intrinsic_motivation.train --gin_files configs/continuous_control/ant/sac_coinflip.gin --base_dir dopamine_logs/d4rl/ant/sac_coinflip/full_runs" +gin-arg ContinuousRunner.num_iterations 400 400 400 400 400 400 400 400 400 +tag --sub_dir
onager prelaunch +jobname ant_coinflip_full +command "XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=.. python -m bonus_based_exploration.cc_intrinsic_motivation.train --gin_files configs/continuous_control/ant/sac_coinflip.gin --base_dir dopamine_logs/d4rl/ant/sac_coinflip/full_runs" +gin-arg ContinuousRunner.num_iterations 400 400 400 400 400 400 400 400 400 +tag --sub_dir
onager prelaunch +jobname ant_coinflip_full +command "XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=.. python -m bonus_based_exploration.cc_intrinsic_motivation.train --gin_files configs/continuous_control/ant/sac_coinflip.gin --base_dir dopamine_logs/d4rl/ant/sac_coinflip/full_runs" +gin-arg ContinuousRunner.num_iterations 400 400 400 400 400 400 400 400 400 +tag --sub_dir

# Gridworld stochasticity sweep
onager prelaunch +jobname gridworld_coinflip_stochastity_sweep +command "PYTHONPATH=.. python -m bonus_based_exploration.train --gin_files configs/gridworld_configs/rainbow_coinflip.gin --base_dir dopamine_logs/gridworld/rainbow_coinflip/42/stochasticity_sweep --gin_bindings 'create_exploration_agent.debug_mode = True'" +gin-arg GridWorldEnv.action_noise_prob 0.0 0.1 0.2 0.3 0.4 0.5 +gin-arg GridWorldEnv.size 42 42 42 42 42 42 42 42 42 42 +tag --sub_dir
onager prelaunch +jobname gridworld_rnd_stochastity_sweep +command "PYTHONPATH=.. python -m bonus_based_exploration.train --gin_files configs/gridworld_configs/rainbow_rnd.gin --base_dir dopamine_logs/gridworld/rainbow_rnd/42/stochasticity_sweep --gin_bindings 'create_exploration_agent.debug_mode = True'" +gin-arg GridWorldEnv.action_noise_prob 0.0 0.1 0.2 0.3 0.4 0.5 +gin-arg GridWorldEnv.size 42 42 42 42 42 42 42 42 42 42 +tag --sub_dir

# Monte Sticky Action Sweep
onager prelaunch +jobname monte_coinflip_sticky_action_sweep +command "PYTHONPATH=.. python -m bonus_based_exploration.train --gin_files configs/monte_configs/rainbow_coinflip.gin --base_dir dopamine_logs/monte/rainbow_coinflip/sticky_action_sweep --gin_bindings 'create_exploration_agent.debug_mode = True'" +gin-arg Runner.num_iterations 200 200 200 200 200 200 +gin-arg create_environment_functions.create_atari_environment.sticky_action_prob 0.0 0.125 0.25 0.375 0.5 0.625 0.75  +tag --sub_dir
onager prelaunch +jobname monte_rnd_sticky_action_sweep +command "PYTHONPATH=.. python -m bonus_based_exploration.train --gin_files configs/monte_configs/rainbow_rnd.gin --base_dir dopamine_logs/monte/rainbow_rnd/sticky_action_sweep --gin_bindings 'create_exploration_agent.debug_mode = True'" +gin-arg Runner.num_iterations 200 200 200 200 200 200 +gin-arg create_environment_functions.create_atari_environment.sticky_action_prob 0.0 0.125 0.25 0.375 0.5 0.625 0.75  +tag --sub_dir