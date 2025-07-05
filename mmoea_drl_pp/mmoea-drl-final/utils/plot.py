import matplotlib.pyplot as plt

def plot_pareto_front(objectives, title="Pareto Front", save_path=None):
    """
    Args:
        objectives: List of tuples [(f1, f2), ...]
        title: Plot title
        save_path: If given, saves the plot as PNG
    """
    f1 = [obj[0] for obj in objectives]
    f2 = [obj[1] for obj in objectives]

    plt.figure(figsize=(8, 6))
    plt.scatter(f1, f2, c="blue", label="Final Solutions", alpha=0.7)
    plt.xlabel("Total Path Length (f1)")
    plt.ylabel("Max Time (f2)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
