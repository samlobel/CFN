import numpy as np
import matplotlib.pyplot as plt


def truncate(scores, max_length=-1, min_length=-1):
    filtered_scores = [score_list for score_list in scores if len(score_list) > min_length]
    if not filtered_scores:
        return filtered_scores
    min_length = min([len(x) for x in filtered_scores])
    if max_length > 0:
        min_length = min(min_length, max_length)
    truncated_scores = [score[:min_length] for score in filtered_scores]
    
    return truncated_scores


def get_plot_params(array):
    median = np.median(array, axis=0)
    means = np.mean(array, axis=0)
    std = np.std(array, axis=0)
    N = array.shape[0]
    top = means + (std / np.sqrt(N))
    bot = means - (std / np.sqrt(N))
    return median, means, top, bot


def moving_average(a, n=25):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n-1:] / n


def smoothen_data(scores, n=10):
    smoothened_cols = scores.shape[1] - n + 1
    smoothened_data = np.zeros((scores.shape[0], smoothened_cols))
    for i in range(scores.shape[0]):
        smoothened_data[i, :] = moving_average(scores[i, :], n=n)
    return smoothened_data


def generate_plot(score_array, label, smoothen=0, linewidth=2, all_seeds=False):
    # smoothen is a number of iterations to average over
    if smoothen > 0:
        score_array = smoothen_data(score_array, n=smoothen)
    median, mean, top, bottom = get_plot_params(score_array)
    plt.plot(mean, linewidth=linewidth, label=label, alpha=0.9)
    plt.fill_between( range(len(top)), top, bottom, alpha=0.2 )
    if all_seeds:
        for i, score in enumerate(score_array):
            plt.plot(score, linewidth=linewidth, label=label+f"_{i+1}", alpha=0.6)
