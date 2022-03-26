from matplotlib import pyplot as plt


def plot_loss(loss_list, iso_time, version, iteration) -> None:
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5), dpi=150)

    if version == "current":
        ax.set_title(f"Decision Transformer Cross-entropy Loss for iteration {iteration}")
        filename = f"loss_for_iter_{iteration:02}"
        ax.set_ylim([0, 2])
    elif version == "detail":
        ax.set_title(f"Decision Transformer Cross-entropy Loss for iteration {iteration}")
        filename = f"loss_for_iter_{iteration:02}_detail"
    elif version == "so far":
        ax.set_title(f"Decision Transformer Cross-entropy Loss after {iteration} iteration(s)")
        filename = f"loss_after_iter_{iteration:02}"
    else:
        raise ValueError(f"Invalid version argument {version} for plot function")

    ax.set_xlabel("Iteration Steps")
    ax.set_ylabel("Loss")

    ax.set_xlim([0, len(loss_list) - 1])

    ax.plot(loss_list)

    fig.tight_layout()
    fig.savefig(f"plots/{iso_time}/{filename}.png")
    plt.close(fig)


if __name__ == "__main__":
    import os
    import time
    from datetime import datetime

    losses = [1.9, 1.8, 1.8, 1.83, 1.7]
    t = datetime.fromtimestamp(time.time()).replace(microsecond=0).isoformat()

    os.mkdir(f"plots/{t}/")

    plot_loss(losses, iso_time=t, iteration=0)
