import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from submission.meta_game_experiments.meta_solver_exploitability import (
    META_POLICY_SAVE_PATHS
)

EPSILON = 1e-6
ONE_COL = True
JOINT_POLICIES = True


def main():
    prefix, files_data = _get_inputs()
    files_to_process = _preprocess_inputs(prefix, files_data)

    all_meta_policies = []
    all_actions_names = []
    all_welfares = []
    all_titles = []
    for i, (file_path, file_data) in enumerate(
        zip(files_to_process, files_data)
    ):
        meta_policies, welfares = _get_policies(file_path)
        actions_names = _get_actions_names(welfares, i)
        all_meta_policies.append(meta_policies)
        all_actions_names.append(actions_names)
        all_welfares.append(welfares)
        all_titles.append(file_data[0])
    assert len(set([str(el) for el in all_actions_names])) == 1
    assert len(set(all_welfares)) == 1

    all_titles = [ el.replace("Alpha", r"$\alpha$") for el in all_titles]

    plot_policies(
        all_meta_policies,
        all_actions_names[0],
        all_titles=all_titles,
        one_col=ONE_COL,
    )


def _get_inputs():
    prefix = os.path.join(
        os.path.dirname(__file__),
        "../meta_game_experiments/results",
    )
    files_data = [
        (
            k,
            os.path.join(
                os.path.split(v)[0],
                "final_base_game/final_eval_in_base_game.json",
            ),
        )
        for k, v in META_POLICY_SAVE_PATHS.items()
    ]
    return prefix, files_data


def _preprocess_inputs(prefix, files_data):
    files_to_process = [
        os.path.expanduser(os.path.join(prefix, file_data[1]))
        for file_data in files_data
    ]
    return files_to_process


def _get_policies(file_path):
    parent_dir, _ = os.path.split(file_path)
    parent_parent_dir, _ = os.path.split(parent_dir)
    meta_policies_file = os.path.join(parent_parent_dir, "meta_policies.json")
    with (open(meta_policies_file, "rb")) as f:
        meta_policies = json.load(f)
    meta_policies = meta_policies["meta_policies"]
    print("meta_policies", type(meta_policies), meta_policies)

    parent_parent_parent_dir, _ = os.path.split(parent_parent_dir)
    welfares_file = os.path.join(
        parent_parent_parent_dir, "payoffs_matrices_0.json"
    )
    with (open(welfares_file, "rb")) as f:
        welfares = json.load(f)
    welfares = welfares["welfare_fn_sets"]
    print("welfares", type(welfares), welfares)

    return meta_policies, welfares


def _get_actions_names(welfares, player_idx):
    actions_names = (
        welfares.replace("OrderedSet", "")
        .replace("(", "")
        .replace(")", "")
        .replace("[", "")
        .lstrip()
    )
    actions_names = actions_names.split("],")
    actions_names = [f"({el})" for el in actions_names]
    actions_names = [
        el.replace("]", "").replace(" ", "") for el in actions_names
    ]
    actions_names = [
        el.replace("'mixed'", r"w^{Nash}") for el in actions_names
    ]
    actions_names = [
        el.replace("'egalitarian'", r"w^{IA}") for el in actions_names
    ]
    actions_names = [
        el.replace("'utilitarian'", r"w^{Util}") for el in actions_names
    ]
    actions_names = [el.replace(",", ", ") for el in actions_names]
    actions_names = [
        el.replace("(", "\{").replace(")", "\}") for el in actions_names
    ]

    default_pi = r"π_1^{Util}"
    actions_names_player_1 = [f"${el}, {default_pi}$" for el in actions_names]
    default_pi = r"π_2^{IA}"
    actions_names_player_2 = [f"${el}, {default_pi}$" for el in actions_names]

    return [actions_names_player_1, actions_names_player_2]


def plot_policies(
    all_meta_policies,
    actions_names,
    all_titles=None,
    one_col=False,
    path_prefix="",
):

    plt.style.use("default")

    if one_col:
        fig, (ax, ax2, ax3, ax4, ax5, ax6) = plt.subplots(
            6, 1, figsize=(10, 15)
        )
    else:
        fig, ((ax, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(
            3, 2, figsize=(20, 10)
        )
    all_ax = [ax, ax2, ax3, ax4, ax5, ax6]

    for i, (meta_policies, ax_in_use, title_) in enumerate(
        zip(all_meta_policies, all_ax, all_titles)
    ):

        # if len(actions_names) != 2:
        #     actions_names = (actions_names, actions_names)
        print("actions_names", actions_names, "len", len(actions_names[0]))
        policies_p0 = []
        policies_p1 = []
        for meta_policy in meta_policies:
            policies_p0.append(meta_policy["player_row"])
            policies_p1.append(meta_policy["player_col"])
        print("policies_p0", len(policies_p0), "policies_p1", len(policies_p1))
        policies_p0 = np.array(policies_p0)
        policies_p1 = np.array(policies_p1)

        if title_ is not None:
            ax_in_use.set_title(title_, fontweight="bold", fontsize=16)
        if JOINT_POLICIES:
            im, cbar = _plot_joint_policies_vanilla(
                policies_p0,
                policies_p1,
                actions_names,
                ax_in_use,
                plot_x_labels=(i > 3 and not one_col) or (i == 5 and one_col),
                plot_y_labels=i % 2 == 0 or one_col,
                full_black=i == 5 or i == 2,
            )
        else:
            im, cbar = _plot_means(
                policies_p0,
                policies_p1,
                actions_names,
                ax_in_use,
                plot_x_labels=(i > 3 and not one_col) or (i == 5 and one_col),
                plot_y_labels=i % 2 == 0 or one_col,
                full_black=i == 5 or i == 2,
            )

    cbaxes = fig.add_axes([0.90, 0.25, 0.025, 0.5])
    cb = plt.colorbar(im, cax=cbaxes)
    cb.ax.set_ylabel(
        "Averaged probability", rotation=-90, va="bottom", fontsize=20
    )

    plt.tight_layout(rect=[0.0, 0.0, 0.85, 1.0])
    path_prefix = os.path.expanduser(path_prefix)
    plt.savefig(f"{path_prefix}meta_policies.png")

    plt.style.use("seaborn-whitegrid")


def _plot_means(
    policies_p0,
    policies_p1,
    actions_names,
    ax,
    plot_x_labels=True,
    plot_y_labels=True,
    full_black=False,
):
    policies_p0_mean = policies_p0.mean(axis=0)
    policies_p1_mean = policies_p1.mean(axis=0)
    policies_mean = np.stack([policies_p0_mean, policies_p1_mean], axis=0)
    policies_p0_std = policies_p0.std(axis=0)
    policies_p1_std = policies_p1.std(axis=0)
    policies_std = np.stack([policies_p0_std, policies_p1_std], axis=0)
    im, cbar = heatmap(
        policies_mean,
        ["player_row", "player_col"],
        actions_names[0],
        ax=ax,
        cmap="YlGn",
        cbarlabel="MEAN proba",
        vmin=0.0,
        vmax=1.0,
        plot_color_scale=False,
        plot_x_labels=plot_x_labels,
        plot_y_labels=plot_y_labels,
        aspect="auto",
    )
    if full_black:
        texts = annotate_heatmap_wt_std(
            im,
            std=policies_std,
            valfmt="{x:.2f}",
            textcolors=("black", "black"),
        )
    else:
        texts = annotate_heatmap_wt_std(im, std=policies_std, valfmt="{x:.2f}")
    return im, cbar


def _plot_joint_policies_vanilla(
    policies_p0,
    policies_p1,
    actions_names,
    ax,
    plot_x_labels,
    plot_y_labels,
    full_black,
):
    policies_p0 = np.expand_dims(policies_p0, axis=-1)
    policies_p1 = np.expand_dims(policies_p1, axis=-1)
    policies_p1 = np.transpose(policies_p1, (0, 2, 1))
    joint_policies = np.matmul(policies_p0, policies_p1)
    return _plot_joint_policies(
        joint_policies,
        actions_names,
        ax,
        plot_x_labels,
        plot_y_labels,
        full_black,
    )


def _plot_joint_policies(
    joint_policies,
    actions_names,
    ax,
    plot_x_labels=True,
    plot_y_labels=True,
    full_black=False,
):
    assert np.all(
        np.abs(joint_policies.sum(axis=2).sum(axis=1) - 1.0) < EPSILON
    ), f"{np.abs(joint_policies.sum(axis=2).sum(axis=1) - 1.0)}"
    joint_policies_mean = joint_policies.mean(axis=0)

    from matplotlib import cm

    # print(cm)
    color_map = cm.get_cmap("gist_heat").reversed()

    im, cbar = heatmap(
        joint_policies_mean,
        actions_names[0],
        actions_names[1],
        ax=ax,
        cmap=color_map,  # "gist_heat",  # "YlGn",
        cbarlabel="Joint policy",
        vmin=0.0,
        vmax=1.0,
        plot_color_scale=False,
        plot_x_labels=plot_x_labels,
        plot_y_labels=plot_y_labels,
        aspect="auto",
    )

    joint_policies_std = joint_policies.std(axis=0)
    if full_black:
        texts = annotate_heatmap_wt_std(
            im,
            std=joint_policies_std,
            valfmt="{x:.2f}",
            textcolors=("black", "black"),
        )
    else:
        texts = annotate_heatmap_wt_std(
            im, std=joint_policies_std, valfmt="{x:.2f}"
        )
    return im, cbar


def heatmap(
    data,
    row_labels,
    col_labels,
    ax=None,
    cbar_kw={},
    cbarlabel="",
    plot_color_scale=True,
    plot_x_labels=True,
    plot_y_labels=True,
    **kwargs,
):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    if plot_color_scale:
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    else:
        cbar = None

    # We want to show all ticks...
    # ... and label them with the respective list entries.
    if plot_x_labels:
        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_xticklabels(col_labels, fontsize=12)
    else:
        ax.set_xticks([])
        ax.set_xticklabels([])
    if plot_y_labels:
        ax.set_yticks(np.arange(data.shape[0]))
        ax.set_yticklabels(row_labels, fontsize=12)
    else:
        ax.set_yticks([])
        ax.set_yticklabels([])

    # Let the horizontal axes labeling appear on top.
    # ax.tick_params(
    #     top=True,
    #     bottom=False,
    #     labeltop=True,
    #     labelbottom=False,
    # )

    # Rotate the tick labels and set their alignment.
    plt.setp(
        ax.get_xticklabels(), rotation=20, ha="right", rotation_mode="anchor"
    )

    # Turn spines off and create white grid.
    # ax.spines[:].set_visible(False)
    for k, v in ax.spines.items():
        ax.spines[k].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    return im, cbar


def annotate_heatmap_wt_std(
    im,
    std,
    data=None,
    valfmt="{x:.2f}",
    textcolors=("black", "white"),
    threshold=None,
    ha=None,
    **textkw,
):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.0

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(
        horizontalalignment=ha if ha is not None else "center",
        verticalalignment="center",
    )
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    print("data.shape", data.shape)
    print("std.shape", std.shape)
    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])

            text = im.axes.text(
                j,
                i,
                valfmt(data[i, j], None) + " ±" + valfmt(std[i, j], None),
                **kw,
            )
            texts.append(text)

    return texts


if __name__ == "__main__":
    main()
