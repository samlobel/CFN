import matplotlib.pyplot as plt
import seaborn as sns

from .plotting_utils import *
from .comparison_plotting import *
import numpy as np
from .monte_sticky_actions_barplot import make_whole_plot as make_monte_bar_plot
sns.set_style("whitegrid")
sns.set_context("poster")

colorblind_palette = sns.color_palette("colorblind")
PALETTE = [colorblind_palette[i] for i in [3,1,2,0,4,5,6,7]]

sns.set_palette(PALETTE)


BASE_LOG_DIR = "./dopamine_logs"
BASE_COMPILED_PLOT_DIR = "./compiled_plots"

def make_monte_plot_final():
    plt.rcParams["figure.figsize"] = (20,12)
    experiment_name_rainbow = f"{BASE_LOG_DIR}/combined_dopamine_logs/dopamine_logs/monte/rainbow_pixelcnn/full_runs/200m/2xlr_fixed_config_for_seeds"
    experiment_name_pixelcnn = f"{BASE_LOG_DIR}/combined_dopamine_logs/dopamine_logs/monte/rainbow_pixelcnn/full_runs/200m/2xlr_fixed_config_for_seeds"
    experiment_name_rnd = f"{BASE_LOG_DIR}/combined_dopamine_logs/dopamine_logs/monte/rainbow_rnd/full_runs/200m/2xlr_original_config_for_seeds"
    experiment_name_cfn = f"{BASE_LOG_DIR}/combined_dopamine_logs/dopamine_logs/monte/rainbow_coinflip/full_runs/200m/bigmem_qlr2x_1"
    plot_comparison_learning_curves(experiment_name_rainbow, stat='eval_average_return', group_func=lambda x: "Rainbow", truncate_min_length=100, smoothen=3, show=False, linewidth=8)
    plot_comparison_learning_curves(experiment_name_pixelcnn, stat='eval_average_return', group_func=lambda x: "PixelCNN", truncate_min_length=100, smoothen=3, show=False, linewidth=8)
    plot_comparison_learning_curves(experiment_name_rnd, stat='eval_average_return', group_func=lambda x: "RND", truncate_min_length=1, smoothen=3, show=False, linewidth=8)
    plot_comparison_learning_curves(experiment_name_cfn, stat='eval_average_return', group_func=lambda x: "CFN", truncate_min_length=1, smoothen=3, show=False, linewidth=8)
    order = 3, 2, 1, 0
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc="upper left", prop={'size': 36})

    plt.title("Montezuma's Revenge", fontsize=48)
    plt.xlabel("Frames (1e6)", size=36)
    plt.ylabel("Average Return", size=36)
    plt.xticks(size=36)
    plt.yticks(size=36)
    plt.savefig(f"{BASE_COMPILED_PLOT_DIR}/monte/monte_learning_curves.png", bbox_inches='tight')
    plt.close()
    # plt.show()

def make_gridworld_results_curve():
    stat="eval_average_return"
    plt.rcParams["figure.figsize"] = (20,12)
    experiment_name_cfn = os.path.join(BASE_LOG_DIR, 'dopamine_logs/gridworld/rainbow_coinflip/42/full_runs')
    experiment_name_pixelcnn = os.path.join(BASE_LOG_DIR, 'dopamine_logs/gridworld/rainbow_pixelcnn/42/full_runs')
    experiment_name_rnd = os.path.join(BASE_LOG_DIR, 'dopamine_logs/gridworld/rainbow_rnd/42/full_runs')
    experiment_name_vanilla = os.path.join(BASE_LOG_DIR, 'dopamine_logs/gridworld/rainbow_vanilla/42/full_runs')

    plot_comparison_learning_curves(experiment_name_vanilla, group_func=lambda x: "Rainbow", stat=stat, show=False, linewidth=8, truncate_length=250)
    plot_comparison_learning_curves(experiment_name_pixelcnn, group_func=lambda x: "PixelCNN", stat=stat, show=False, linewidth=8, truncate_length=250)
    plot_comparison_learning_curves(experiment_name_rnd, group_func=lambda x: "RND", stat=stat, show=False, linewidth=8, truncate_length=250)
    plot_comparison_learning_curves(experiment_name_cfn, stat=stat, group_func=lambda x: "CFN", truncate_min_length=100, show=False, linewidth=8, truncate_length=250)
    order = 3, 2, 1, 0
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc="upper left", prop={'size': 36})
    plt.title("Gridworld (42x42)", fontsize=48)
    plt.xlabel("Frames (1e3)", size=36)
    plt.ylabel("Average Return", size=36)
    plt.xticks(size=36)
    plt.yticks(size=36)
    plt.savefig(f"{BASE_COMPILED_PLOT_DIR}/gridworld/gridworld_full_comparison.png", bbox_inches='tight')
    plt.close()
    # plt.show()

def _make_various_fetch_plots(task, mode, show=True):
    experiment_name_cfn = os.path.join(BASE_LOG_DIR, f"fetch/{task}/sac_coinflip/mode_{mode}/full_runs")
    experiment_name_rnd = os.path.join(BASE_LOG_DIR, f"fetch/{task}/sac_rnd/mode_{mode}/full_runs")
    experiment_name_vanilla = os.path.join(BASE_LOG_DIR, f"fetch/{task}/sac_vanilla/mode_{mode}/full_runs")
    plot_comparison_learning_curves(experiment_name_cfn, stat='train_average_return', group_func=lambda x: "CFN", truncate_min_length=1, show=False)
    plot_comparison_learning_curves(experiment_name_rnd, stat='train_average_return', group_func=lambda x: "RND", truncate_min_length=1, show=False)
    plot_comparison_learning_curves(experiment_name_vanilla, stat='train_average_return', group_func=lambda x: "SAC", truncate_min_length=1, show=show)

def make_fetch_plot_grid():
    plt.rcParams["figure.figsize"] = (20,12)
    plt.subplot(3,3,1)
    plt.title("Push (Default)")
    _make_various_fetch_plots("push", "-1", False)
    plt.legend()
    plt.subplot(3,3,2)
    plt.title("Push (Medium)")
    _make_various_fetch_plots("push", "1", False)
    plt.subplot(3,3,3)
    plt.title("Push (Hard)")
    _make_various_fetch_plots("push", "2", False)
    plt.subplot(3,3,4)
    plt.title("Slide (Default)")
    _make_various_fetch_plots("slide", "-1", False)
    plt.subplot(3,3,5)
    plt.title("Slide (Medium)")
    _make_various_fetch_plots("slide", "1", False)
    plt.subplot(3,3,6)
    plt.title("Slide (Hard)")
    _make_various_fetch_plots("slide", "2", False)
    plt.subplot(3,3,7)
    plt.title("Pick and Place (Default)")
    _make_various_fetch_plots("pickandplace", "-1", False)
    plt.subplot(3,3,8)
    plt.title("Pick and Place (Medium)")
    _make_various_fetch_plots("pickandplace", "1", False)
    plt.subplot(3,3,9)
    plt.title("Pick and Place (Hard)")
    _make_various_fetch_plots("pickandplace", "2", False)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    # plt.show()
    plt.savefig(f"{BASE_COMPILED_PLOT_DIR}/fetch/fetch_grid.png", bbox_inches='tight')
    plt.close()

def make_gridworld_ablation_curve():
    plt.rcParams["figure.figsize"] = (20,12)
    experiment_name = f"{BASE_LOG_DIR}/gridworld/rainbow_coinflip/42/pri_rp_ablation_full/"
    def group_func_yesyes(name):
        if "userandomprior_True" in name and "useprioritizedbuffer_True" in name:
            return "Random Prior and Prioritization"
    def group_func_yesno(name):
        if "userandomprior_True" in name and "useprioritizedbuffer_False" in name:
            return "Random Prior and No Prioritization"
    def group_func_noyes(name):
        if "userandomprior_False" in name and "useprioritizedbuffer_True" in name:
            return "No Random Prior and Prioritization"
    def group_func_nono(name):
        if "userandomprior_False" in name and "useprioritizedbuffer_False" in name:
            return "No Random Prior and No Prioritization"

    plot_comparison_count_stats(experiment_name=experiment_name, group_func=group_func_nono, plot_mse=True, truncate_min_length=249, show=False, linewidth=8, smoothen=6)
    plot_comparison_count_stats(experiment_name=experiment_name, group_func=group_func_noyes, plot_mse=True, truncate_min_length=249, show=False, linewidth=8, smoothen=6)
    plot_comparison_count_stats(experiment_name=experiment_name, group_func=group_func_yesno, plot_mse=True, truncate_min_length=249, show=False, linewidth=8, smoothen=6)
    plot_comparison_count_stats(experiment_name=experiment_name, group_func=group_func_yesyes, plot_mse=True, truncate_min_length=249, show=False, linewidth=8, smoothen=6)
    order = 3, 2, 1, 0
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc="upper right", prop={'size': 24})

    plt.title("Gridworld (42x42)", fontsize=48)
    plt.xlabel("Frames (1e3)", size=36)
    plt.ylabel("Bonus Prediction Error", size=36)
    plt.xticks(size=36)
    plt.yticks(size=36)

    plt.savefig(f"{BASE_COMPILED_PLOT_DIR}/gridworld/gridworld_ablation_mse.png", bbox_inches='tight')
    plt.close()


def make_gridworld_true_vs_approx_3_subplots():
    plt.rcParams["figure.figsize"] = (42,12)
    experiment_name_cfn = os.path.join(BASE_LOG_DIR, 'dopamine_logs/gridworld/rainbow_coinflip/42/full_runs')
    experiment_name_pixelcnn = os.path.join(BASE_LOG_DIR, 'dopamine_logs/gridworld/rainbow_pixelcnn/42/full_runs')
    experiment_name_rnd = os.path.join(BASE_LOG_DIR, 'dopamine_logs/gridworld/rainbow_rnd/42/full_runs')

    count_dict_filename_cfn = os.oath.join(experiment_name_cfn, 'gridworld_coinflip_full_01__GridWorldEnv.size_42/counts/count_dict.pkl.gz')
    count_dict_filename_pixelcnn = os.oath.join(experiment_name_pixelcnn, 'gridworld_pixelcnn_full_01__GridWorldEnv.size_42/counts/count_dict.pkl.gz')
    count_dict_filename_rnd = os.oath.join(experiment_name_rnd, 'gridworld_rnd_full_01__GridWorldEnv.size_42/counts/count_dict.pkl.gz')
    count_dict_cfn = load_count_dict(count_dict_filename_cfn)
    count_dict_pixelcnn = load_count_dict(count_dict_filename_pixelcnn)
    count_dict_rnd = load_count_dict(count_dict_filename_rnd)

    plt.subplot(1,3,1)
    plot_true_vs_approx_bonus(count_dict_cfn[100], show=False, include_line=True, filter_point=(41, 41), color=PALETTE[3])
    plt.xlabel("True Bonus", size=36)
    plt.ylabel("Approx Bonus", size=36)
    plt.xticks(size=36)
    plt.yticks(size=36)
    plt.subplot(1,3,1)
    plt.title("CFN", fontsize=48)
    ax = plt.subplot(1,3,2)
    plot_true_vs_approx_bonus(count_dict_pixelcnn[100], show=False, include_line=False, filter_point=(41, 41), color=PALETTE[2])
    plt.xlabel("True Bonus", size=36)
    plt.ylabel("", size=36)
    plt.xticks(size=36)
    plt.yticks(size=36)
    plt.title("PixelCNN", fontsize=48)
    pos = ax.get_position()
    offset = 0.01
    print(pos.x0+offset, pos.x1+offset)
    ax.set_position([pos.x0+offset, pos.y0, pos.x1-pos.x0, pos.y1-pos.y0])
    ax = plt.subplot(1,3,3)
    plot_true_vs_approx_bonus(count_dict_rnd[100], show=False, include_line=False, filter_point=(41, 41), color=PALETTE[1])
    plt.xlabel("True Bonus", size=36)
    plt.ylabel("", size=36)
    plt.xticks(size=36)
    plt.yticks(size=36)
    plt.title("RND", fontsize=48)

    pos = ax.get_position()
    offset = 0.005
    print(pos.x0+offset, pos.x1+offset)
    ax.set_position([pos.x0+offset, pos.y0, pos.x1-pos.x0, pos.y1-pos.y0])
    plt.savefig(f"{BASE_COMPILED_PLOT_DIR}/gridworld/subplots_gridworld_all_three_bonus.png", bbox_inches='tight')
    plt.close()
    # plt.show()

# python -m scripts.make_icml_2023_paper_plots
if __name__ == '__main__':
    # make_monte_plot_final()
    # make_gridworld_results_curve()
    # make_gridworld_ablation_curve()
    # make_gridworld_true_vs_approx_3_subplots()
    # make_fetch_plot_grid()
    # make_monte_bar_plot()
    pass
