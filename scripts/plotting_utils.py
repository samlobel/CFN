import os
import glob
import pickle
import gzip
import matplotlib.pyplot as plt
from dopamine.colab import utils as colab_utils
import numpy as np

try:
    import seaborn as sns; sns.set()
except ImportError:
    print("Could not import seaborn. Plotting will not be as pretty.")


def get_summary_data(log_dir, stat):
    raw_data, _ = colab_utils.load_statistics(log_dir, verbose=False)
    summarized_data = colab_utils.summarize_data(raw_data, [stat])
    return summarized_data


def plot_statistic(log_dir, stat='eval_episode_lengths', save_path=None, show=True, legend_name=None):
    # 'train_episode_lengths', 'train_episode_returns', 'train_average_return', 'train_average_steps_per_second', 'eval_episode_lengths', 'eval_episode_returns', 'eval_average_return'
    
    summarized_data = get_summary_data(log_dir, stat)
    plt.plot(summarized_data[stat], label=legend_name)
    plt.plot()
    plt.title(stat)
    plt.xlabel('Iteration')
    plt.ylabel(stat)

    if show:
        plt.legend()
        plt.show()
    
    if save_path is not None:
        plt.legend()
        plt.savefig(save_path)
        plt.close()


def plot_counts(count_dict, save_path=None):
    plt.subplot(1, 2, 1)
    state_count_pairs = [(s, count_dict['true'][s], count_dict['approx'][s]) for s in count_dict['true']]
    states = [x[0] for x in state_count_pairs]
    counts = [x[1] for x in state_count_pairs]
    approx_counts = [1./(x[2]**2) for x in state_count_pairs]
    x = [state[0] for state in states]
    y = [state[1] for state in states]
    plt.scatter(x, y, c=counts)
    plt.title("True counts")
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.scatter(x, y, c=approx_counts)
    plt.colorbar()
    plt.title("Approx counts")
    plt.savefig(save_path) if save_path else plt.show()
    plt.close()

def plot_bonuses(count_dict, save_path=None):
    plt.subplot(1, 2, 1)
    state_count_pairs = [(s, count_dict['true'][s], count_dict['approx'][s]) for s in count_dict['true']]
    states = [x[0] for x in state_count_pairs]
    counts = [x[1] for x in state_count_pairs]
    approx_bonuses = [x[2] for x in state_count_pairs]
    exact_bonuses = [(1. / x[1]) ** 0.5 for x in state_count_pairs]

    x = [state[0] for state in states]
    y = [state[1] for state in states]
    plt.scatter(x, y, c=exact_bonuses)
    plt.title("True bonuses")
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.scatter(x, y, c=approx_bonuses)
    plt.colorbar()
    plt.title("Approx bonuses")
    plt.savefig(save_path) if save_path else plt.show()
    plt.close()


def load_count_dict(file_path):
    opener = gzip.open if file_path.endswith('.gz') else open
    try:
        with opener(file_path, 'rb') as f:
            return pickle.load(f)
    except EOFError:
        print(f"Could not load {file_path}")


def get_true_vs_approx(count_dict, mode, filter_point=None):
    assert mode in ("bonus", "count"), mode
    
    count_pairs = [(count_dict['true'][s], count_dict['approx'][s]) for s in count_dict['true'] if s != filter_point]

    if mode == "count":
        counts = [x[0] for x in count_pairs]
        approx_counts = [1./(x[1]**2) for x in count_pairs]
        return counts, approx_counts
    
    approx_bonus = [x[1] for x in count_pairs]
    exact_bonus = [1./(x[0]**0.5) for x in count_pairs]
    return exact_bonus, approx_bonus


def plot_true_vs_approx_counts(count_dict, save_path=None):
    counts, approx_counts = get_true_vs_approx(count_dict, 'count')
    max_counts = max(counts+approx_counts)

    plt.scatter(counts, approx_counts)
    plt.plot([0, max_counts], [0, max_counts], linestyle="dashed", color="k")
    plt.xlabel("True Counts")
    plt.ylabel("Approx Counts")
    plt.grid()

    plt.savefig(save_path) if save_path else plt.show()
    plt.close()

def _get_r(true, approx):
    return np.corrcoef(true, approx)[0, 1]

def plot_true_vs_approx_bonus(count_dict, save_path=None, show=True, legend_name=None, include_line=True, show_y=True, color=None, alpha=1.0, min_r=float("-inf"), print_num_states=False, filter_point=None, normalize=None):
    # normalize tells you the bounds you want. Sure.
    assert normalize==None or isinstance(normalize, tuple)
    if print_num_states:
        print("num states", len(count_dict['approx']))
    exact_bonus, approx_bonus = get_true_vs_approx(count_dict, 'bonus', filter_point=filter_point)
    if normalize is not None:
        min_approx, max_approx = min(approx_bonus), max(approx_bonus)
        approx_bonus = [(ab - min_approx) / (max_approx - min_approx) for ab in approx_bonus]
    # import ipdb; ipdb.set_trace()
    max_bonus = max(exact_bonus + approx_bonus)
    r_corr = _get_r(exact_bonus, approx_bonus)
    if r_corr < min_r:
        print(f"Skipping, R = {r_corr} < {min_r}")
        return False

    plotted_fig = plt.scatter(exact_bonus, approx_bonus, label=legend_name, color=color, alpha=alpha)
    if include_line:
        plt.plot([0, max_bonus], [0, max_bonus], linestyle="dashed", color="k")
    plt.xlabel("True Bonus")
    if show_y:
        plt.ylabel("Approx Bonus")

    plt.title(f"R = {r_corr:.3f}")

    if show:
        plt.show()
    if save_path:
        plt.savefig(save_path)
        plt.close()

    return plotted_fig

def show_visitation_over_iteration(count_dict, iteration, show=True, grid=True):
    this_iteration_cd = count_dict[iteration]
    last_iteration_cd = count_dict[iteration - 1] if iteration > 0 else {'true': {}}
    states = list(this_iteration_cd['true'].keys())
    valid_states, count_differences = [], []
    for s in states:
        count_this_iteration = this_iteration_cd['true'][s]
        count_last_iteration = last_iteration_cd['true'][s] if s in last_iteration_cd['true'] else 0
        if count_this_iteration - count_last_iteration > 0:
            valid_states.append(s)
            count_differences.append(count_this_iteration - count_last_iteration)

    if grid:
        x = [state[0] for state in valid_states]
        y = [state[1] for state in valid_states]

        # TODO: Maybe make the image look better

        plt.scatter(x, y, c=count_differences, s=400, marker='s', alpha=1.0)
    else:
        plt.scatter(range(len(valid_states)), sorted(count_differences))
    plt.colorbar()
    if show:
        plt.show()


def plot_approx_bonus_vs_counts(count_dict, save_path=None, show=True, legend_name=None, include_line=True, show_y=True, color=None, alpha=1.0, min_r=float("-inf")):
    exact_bonus, approx_bonus = get_true_vs_approx(count_dict, 'bonus')
    exact_count = np.array(exact_bonus) ** -2
    max_bonus = max(approx_bonus)
    r_corr = _get_r(exact_count, approx_bonus)
    if r_corr < min_r:
        print(f"Skipping, R = {r_corr} < {min_r}")
        return False

    plt.scatter(exact_count, approx_bonus, label=legend_name, color=color, alpha=alpha)
    if include_line:
        plt.plot([0, max_bonus], [0, max_bonus], linestyle="dashed", color="k")
    plt.xlabel("True Bonus")
    if show_y:
        plt.ylabel("Approx Bonus")

    plt.title(f"R = {r_corr:.3f}")

    if show:
        plt.show()
    if save_path:
        plt.savefig(save_path)
        plt.close()

    return True


def plot_value_function_gridworld(count_dict, save_path=None):
    state_value_pairs = list(count_dict["value"].items())
    states = [x[0] for x in state_value_pairs]
    values = [x[1] for x in state_value_pairs]

    x = [state[0] for state in states]
    y = [state[1] for state in states]

    # TODO: Maybe make the image look better

    plt.scatter(x, y, c=values, s=2000, marker='s')

    plt.colorbar()
    plt.title("Values")
    plt.savefig(save_path) if save_path else plt.show()
    plt.close()

def plot_spatial_bonus(count_dict, save_path=None, room_num=None, alpha=1.0, min_reward=None, max_reward = None):
    state_bonus_pairs = list(count_dict["approx"].items())
    states = [x[0] for x in state_bonus_pairs]
    approx = [x[1] for x in state_bonus_pairs]

    if room_num is not None:
        length = len(states)
        indices = [i for i, (x,y,r) in enumerate(states) if r == room_num]
        states = [states[i] for i in indices]
        approx = [approx[i] for i in indices]
        print(f'filtered {length} to {len(states)} for room {room_num}')

    if min_reward is not None:
        length = len(states)
        indices = [i for i, bonus in enumerate(approx) if bonus >= min_reward]
        states = [states[i] for i in indices]
        approx = [approx[i] for i in indices]
        print(f'filtered {length} to {len(states)} for min_reward={min_reward}')

    if max_reward is not None:
        length = len(states)
        indices = [i for i, bonus in enumerate(approx) if bonus < max_reward]
        states = [states[i] for i in indices]
        approx = [approx[i] for i in indices]
        print(f'filtered {length} to {len(states)} for max_reward={max_reward}')


    x = [state[0] for state in states]
    y = [state[1] for state in states]

    # TODO: Maybe make the image look better

    plt.scatter(x, y, c=approx, s=400, marker='s', alpha=alpha)

    plt.colorbar()
    plt.title("Approx Counts")
    plt.savefig(save_path) if save_path else plt.show()
    plt.close()


def plot_mse_vs_iteration(full_count_dict, mode="bonus", save_path=None, show=True, legend_name=None):
    assert mode in ("bonus", "count"), mode

    def get_rmse_for_each_iteration(count_dict):
        exact, approx = get_true_vs_approx(count_dict, mode)
        assert len(exact) == len(approx)
        exact = np.asarray(exact)
        approx = np.asarray(approx)
        sq_errors = (exact-approx) ** 2
        root_mean_sq_errors = np.mean(sq_errors) ** 0.5
        return root_mean_sq_errors
    
    iterations = list(sorted(full_count_dict.keys()))
    rmses = [get_rmse_for_each_iteration(full_count_dict[iter]) for iter in iterations]
    
    plt.plot(iterations, rmses, label=legend_name)
    plt.ylabel("RMSE")
    plt.xlabel("Iteration")
    plt.grid()
    if show:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()


def extract_exploration_amounts(full_count_dict):
    iterations = list(sorted(full_count_dict.keys()))
    num_visited_states = [len(full_count_dict[iter]['true']) for iter in iterations]
    return iterations, num_visited_states


def plot_exploration_amounts(full_count_dict, save_path=None, show=True, legend_name=None):
    iterations, num_visited_states = extract_exploration_amounts(full_count_dict)
    
    plt.plot(iterations, num_visited_states, label=legend_name)
    plt.xlabel("Iteration")
    plt.ylabel("Num Visited States")
    plt.grid()

    if show:
        plt.show()
        
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()


def print_rooms_for_runs(experiment_name, episode=-1):
    cdfs = glob.glob(os.path.join(experiment_name, "*", "counts", "count_dict.pkl.gz"), recursive=True)
    for cdf in cdfs:
        cd = load_count_dict(cdf)
        episode_to_print = episode if episode >= 0 else max(cd.keys())
        try:
            print(cdf)
            rooms = cd[episode_to_print]['visited_rooms']
            print(f"Ep: {episode_to_print}   Num Rooms: {len(rooms)}    Rooms: {rooms}")            
        except Exception as e:
            print("Failed")

def do_thing_to_all(experiment_directory, thing_func, iter=-1, filter_func=None, kwarg_dict=dict()):
    cdfs = glob.glob(os.path.join(experiment_directory, "*", "counts", "count_dict.pkl.gz"), recursive=True)
    print(cdfs)
    for cdf in cdfs:
        if filter_func is not None:
            if not filter_func(cdf):
                print('skipping')
                continue
        try:
            cd = load_count_dict(cdf)
            iteration = iter if iter >= 0 else max(cd.keys())
            print(iteration, cdf.split("/")[-3], '\n')
            thing_func(cd[iteration], **kwarg_dict)
        except Exception as e:
            print("Failed for", cdf.split('/')[-3], '\n')
            print(e)
            pass




def plot_all_count_dicts(experiment_directory, iter=-1, filter_func=None, kwarg_dict=dict()):
    do_thing_to_all(experiment_directory, plot_true_vs_approx_bonus, iter, filter_func, kwarg_dict=kwarg_dict)

def plot_all_spatial_bonuses(experiment_directory, iter=-1, filter_func=None, kwarg_dict=dict()):
    do_thing_to_all(experiment_directory, plot_spatial_bonus, iter, filter_func, kwarg_dict=kwarg_dict)




if __name__ == "__main__":
    # Examle plotting
    count_dict_path = "./dopamine_logs/gridworld/rainbow_coinflip/42/full_runs/gridworld_coinflip_full_01__GridWorldEnv.size_42/counts/count_dict.pkl"
    count_dict = load_count_dict(count_dict_path)
    plot_exploration_amounts(count_dict, show=True, legend_name='RainbowCoinFlip')
