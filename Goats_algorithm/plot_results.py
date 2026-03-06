import matplotlib.pyplot as plt


def plot_sweeping_currency(steps, currency, types, dpi=200, savepath="sweeping_plot.png"):
    plt.figure(dpi=dpi)

    steps_forward = [s for s, t in zip(steps, types) if t == "-->"]
    curr_forward = [c for c, t in zip(currency, types) if t == "-->"]

    steps_backward = [s for s, t in zip(steps, types) if t == "<--"]
    curr_backward = [c for c, t in zip(currency, types) if t == "<--"]

    plt.scatter(
        steps_forward,
        curr_forward,
        color="orange",
        label="--> (forward)"
    )

    plt.scatter(
        steps_backward,
        curr_backward,
        color="green",
        label="<-- (backward)"
    )

    plt.plot(steps, currency, color="gray", alpha=0.3)

    plt.xlabel("Sites")
    plt.ylabel("Currency")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()

    print(f"Plot salvato in: {savepath}")