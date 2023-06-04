from scripts.plotting_utils import *
from scripts.comparison_plotting import *
import glob
import os
from dopamine.colab import utils as colab_utils
import seaborn as sns

BASE_LOG_DIR = "./dopamine_logs"
BASE_COMPILED_PLOT_DIR = "./compiled_plots"

colorblind_palette = sns.color_palette("colorblind")
PALETTE = [colorblind_palette[i] for i in [3,1,2,0,4,5,6,7]]


def get_cumulative_reward_for_run(path, stat='train_average_return'):
    LENGTH = 100
    data = get_summary_data(path, stat=stat)
    data = data[stat]
    data = data[:LENGTH]
    if len(data) != LENGTH:
        print(f'skipping length {len(data)}')
        return None
    assert len(data) == LENGTH, len(data)
    return np.mean(data)

def get_matching_paths_cfn(sticky_action_prob):
    assert isinstance(sticky_action_prob, str)
    experiment_name_monte_cfn_transitionnoise_sweep = os.path.join(BASE_LOG_DIR, "monte/rainbow_coinflip/sticky_action_sweep")
    matching_paths = glob.glob(os.path.join(experiment_name_monte_cfn_transitionnoise_sweep, f"*stickyactionprob_{sticky_action_prob}*", "logs"), recursive=True)
    return matching_paths

def get_matching_paths_rnd(sticky_action_prob, reward_scale=None):
    assert isinstance(sticky_action_prob, str)
    experiment_name_monte_rnd_transitionnoise_sweep = os.path.join(BASE_LOG_DIR, "monte/rainbow_rnd/sticky_action_sweep")
    matching_paths = glob.glob(os.path.join(experiment_name_monte_rnd_transitionnoise_sweep, f"*stickyactionprob_{sticky_action_prob}*", "logs"), recursive=True)
    return matching_paths

def make_data_dict_for_all_noise_levels(mode='cfn'):
    assert mode in ['cfn', 'rnd']
    if mode == 'cfn':
        match_function = get_matching_paths_cfn
    elif mode == 'rnd':
        match_function = get_matching_paths_rnd
    else:
        raise ValueError(mode)

    data_dict = {}
    for noise_prob in ["0.0", "0.125", "0.25", "0.375", "0.5", "0.625", "0.75"]:
        matching_paths = match_function(noise_prob)
        data_dict[noise_prob] = []
        for matching_path in matching_paths:
            data = get_cumulative_reward_for_run(matching_path)
            if data is None:
                print(f'skipping seed for {mode} ({noise_prob})')
            else:
                data_dict[noise_prob].append(data)
    return data_dict

def make_cfn_plot_err():
    offset = -0.02
    width = 0.04
    color=sns.color_palette(PALETTE)[3]
    data_dict = make_data_dict_for_all_noise_levels(mode='cfn')
    noise_probs = [p for p in data_dict.keys()]
    reward_means = [np.mean(data_dict[x]) for x in noise_probs]
    reward_std_err = [np.std(data_dict[x])/len(data_dict[x])**0.5 for x in noise_probs]
    noise_probs = [float(p) for p in noise_probs]
    x_axis = [p + offset for p in noise_probs]
    plt.bar(x=x_axis, height=reward_means, yerr=reward_std_err, label="CFN", width=width, color=color)

def make_rnd_plot_err():
    offset = 0.02
    width = 0.04
    color=sns.color_palette(PALETTE)[2]
    data_dict = make_data_dict_for_all_noise_levels(mode='rnd') # reward-scale is ignored.
    noise_probs = [p for p in data_dict.keys()]
    reward_means = [np.mean(data_dict[x]) for x in noise_probs]
    reward_std_err = [np.std(data_dict[x])/len(data_dict[x])**0.5 for x in noise_probs]
    noise_probs = [float(p) for p in noise_probs]
    x_axis = [p + offset for p in noise_probs]
    plt.bar(x=x_axis, height=reward_means, yerr=reward_std_err, label="RND", width=width, color=color)


def make_whole_plot():
    import seaborn as sns
    sns.set_style("whitegrid")
    sns.set_context("poster")
    sns.set_palette(PALETTE)
    plt.rcParams["figure.figsize"] = (20,12)
    make_cfn_plot_err()
    make_rnd_plot_err()
    plt.xlabel("Sticky Action Probability", size=36)
    plt.ylabel("Mean Episodic Return", size=36)
    plt.xticks(size=36)
    plt.yticks(size=36)
    plt.legend(prop={'size': 36})
    plt.title("Montezuma's Revenge", fontsize=48)
    # plt.show()
    plt.savefig(f"{BASE_COMPILED_PLOT_DIR}/monte/monte_sticky_action_barplot_updated.png",bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    make_whole_plot()
