import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from .comparison_plotting_utils import *
from .plotting_utils import get_summary_data, extract_exploration_amounts, load_count_dict, get_true_vs_approx


def get_config(log_dir_name, group_key):
    try:
        return re.search(f".*({group_key}_[^_]*)_.*", log_dir_name).group(1)
    except:
        return re.search(f".*({group_key}_[^_]*)/.*", log_dir_name).group(1)


def default_make_key(log_dir_name, group_keys):
    keys = [get_config(log_dir_name, group_key) for group_key in group_keys]
    key = "_".join(keys)
    return key


def extract_log_dirs(dir_path, stat, group_keys=("rewardscale",)):

    # Map config to a list of curves
    log_dir_map = defaultdict(list)

    for log_dir in glob.glob(dir_path, recursive=True):
        try:
            keys = [get_config(log_dir, group_key) for group_key in group_keys]
            key = "_".join(keys)
            key = default_make_key(log_dir, group_keys)
            # key = get_config(log_dir, group_key)
            curve = get_summary_data(log_dir, stat)[stat]
            log_dir_map[key].append(curve)
        except:
            print(f"Could not extract {stat} from {log_dir}")

    return log_dir_map

def extract_log_dirs_group_func(dir_path, stat, group_func=lambda x: x):

    # Map config to a list of curves
    log_dir_map = defaultdict(list)

    for log_dir in glob.glob(dir_path, recursive=True):
        try:
            key = group_func(log_dir)
            if key is None:
                continue
            # key = get_config(log_dir, group_key)
            curve = get_summary_data(log_dir, stat)[stat]
            log_dir_map[key].append(curve)
        except:
            print(f"Could not extract {stat} from {log_dir}")

    return log_dir_map


def extract_log_dirs_new(path_dict, stat):
    log_dir_map = defaultdict(list)
    for key, val in path_dict.items():
        for fname in val:
            try:
                sum_data = get_summary_data(fname, stat)[stat]
                print(fname, len(sum_data))
                log_dir_map[key].append(get_summary_data(fname, stat)[stat])
            except:
                print('tried')
    
    return log_dir_map


def make_name_from_group_keys(name, group_keys):
    keys = [get_config(name, group_key) for group_key in group_keys]
    return "_".join(keys)

def extract_count_dicts(dir_path, group_keys=("rewardscale",)):

    # Map config to a list of count_dicts
    log_dir_map = defaultdict(list)

    for path_to_dict in glob.glob(dir_path, recursive=True):
        keys = [get_config(path_to_dict, group_key) for group_key in group_keys]
        key = "_".join(keys)
        print(f"grouping {path_to_dict} to {key}")
        count_dict = load_count_dict(path_to_dict)
        if count_dict is not None:
            log_dir_map[key].append(count_dict)
    
    return log_dir_map

def extract_count_dicts_group_func(dir_path, group_func=lambda x: x):
    # Map config to a list of count_dicts
    log_dir_map = defaultdict(list)

    for path_to_dict in glob.glob(dir_path, recursive=True):
        key = group_func(path_to_dict)
        if key is None:
            continue
        print(f"grouping {path_to_dict} to {key}")
        count_dict = load_count_dict(path_to_dict)
        if count_dict is not None:
            log_dir_map[key].append(count_dict)
    
    return log_dir_map


def plot_comparison_learning_curves(
    experiment_name=None,
    stat='eval_episode_lengths',
    group_keys=("rewardscale",),
    group_func=None,
    save_path=None,
    show=True,
    smoothen=10,
    log_dir_path_map=None,
    uniform_truncate=False,
    truncate_length=-1,
    truncate_min_length=-1,
    ylabel=False,
    legend_loc=None,
    linewidth=2,
    min_seeds=1,
    all_seeds=False,
    title=None,
    min_final_val=None):

    # import seaborn as sns
    # NUM_COLORS=100
    # clrs = sns.color_palette('husl', n_colors=NUM_COLORS)
    # sns.set_palette(clrs)

    assert isinstance(group_keys, (tuple, list)), f"{type(group_keys)} should be tuple or list"
    if save_path is not None:
        plt.figure(figsize=(24,12))


    ylabel = ylabel or stat

    dir_path = os.path.join(experiment_name, "*/logs")

    if log_dir_path_map is None:
        if group_func is not None:
            log_dir_path_map = extract_log_dirs_group_func(dir_path, stat=stat, group_func=group_func)
        else:
            log_dir_path_map = extract_log_dirs(dir_path, stat=stat, group_keys=group_keys)

    for config in log_dir_path_map:
        if config is None:
            continue
        curves = log_dir_path_map[config]
        print(config)
        for curve in curves:
            print(f"\t{len(curve)}")
        truncated_curves = truncate(curves, max_length=truncate_length, min_length=truncate_min_length)
        if len(truncated_curves) < min_seeds:
            continue

        if min_final_val is not None:
            if np.array(truncated_curves)[:, -1].mean() <= min_final_val:
                continue

        score_array = np.array(truncated_curves)
        print(np.max(score_array))
        generate_plot(
            score_array,
            label=config,
            smoothen=smoothen,
            linewidth=linewidth,
            all_seeds=all_seeds)
    
    # plt.grid()
    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    if title:
        plt.title(title)

    if show:
        if legend_loc:
            plt.legend(loc=legend_loc)
        else:
            plt.legend()
        plt.show()
    
    if save_path is not None:
        plt.legend()
        plt.savefig(save_path)
        plt.close()


def get_rmse_for_each_iteration(count_dict):
    exact, approx = get_true_vs_approx(count_dict, "bonus")
    assert len(exact) == len(approx)
    exact = np.asarray(exact)
    approx = np.asarray(approx)
    sq_errors = (exact-approx) ** 2
    root_mean_sq_errors = np.mean(sq_errors) ** 0.5
    return root_mean_sq_errors


# TODO: Generalize to all count stats
def plot_comparison_count_stats(
    experiment_name,
    save_path=None,
    show=True,
    stat=None,
    group_keys=("rewardscale",),
    group_func=None,
    smoothen=10,
    truncate_min_length=-1,
    plot_mse=False,
    plot_rooms=False,
    min_seeds=1,
    linewidth=2,
    all_seeds=False,
    title=None):

    assert isinstance(group_keys, (tuple, list)), f"{type(group_keys)} should be tuple or list"

    dir_path = os.path.join(experiment_name, "*/counts/count_dict.pkl*")

    if group_func is not None:
        grouped_count_dicts = extract_count_dicts_group_func(dir_path, group_func=group_func)
    else:
        grouped_count_dicts = extract_count_dicts(dir_path, group_keys=group_keys)

    for config in grouped_count_dicts:
        if config is None:
            continue
        count_dicts = grouped_count_dicts[config]

        if plot_mse:
            curves = [[get_rmse_for_each_iteration(cd[iteration]) for iteration in cd.keys()] for cd in count_dicts]
        elif plot_rooms:
            curves = [[len(cd[iteration]['visited_rooms']) for iteration in cd] for cd in count_dicts]
        else:
            curves = [extract_exploration_amounts(cd)[1] for cd in count_dicts]
        # curves = [c for c in curves if c is not None]

        truncated_curves = truncate(curves, min_length=truncate_min_length)
        if len(truncated_curves) < min_seeds:
            continue
        score_array = np.array(truncated_curves)
        generate_plot(
            score_array, label=config, smoothen=smoothen,
            all_seeds=all_seeds, linewidth=linewidth)
    
    # plt.grid()
    plt.xlabel("Iteration")
    if plot_mse:
        plt.ylabel('MSE')
    elif plot_rooms:
        plt.ylabel('rooms visited')
    else:
        plt.ylabel("# States Visited")
    if title:
        plt.title(title)

    if show:
        plt.legend()
        plt.show()
    
    if save_path is not None:
        plt.legend()
        plt.savefig(save_path)
        plt.close()


if __name__ == "__main__":
    # Examle plotting
    experiment_name = "./dopamine_logs/gridworld/rainbow_rnd/42/stochasticity_sweep/"
    plot_comparison_learning_curves(
        experiment_name,
        stat='eval_average_return',
        save_path=None,
        show=True,
        group_keys=("actionnoiseprob",),
        group_func=None,
        # smoothen=10,
        smoothen=False,
        truncate_min_length=10,
        # min_seeds=5,
        all_seeds=False,
        title="Gridworld Stochasticity Sweep Learning Curves"
        )
