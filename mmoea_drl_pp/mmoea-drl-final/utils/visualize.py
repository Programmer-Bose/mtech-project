import matplotlib.pyplot as plt
from drl import planner_interface  # Import the module, not just variables

def plot_best_solution(individual, title="Best Solution", save_path=None):
    """
    Plots the task paths for each robot in a multi-depot system.
    
    Args:
        individual: List[List[int]] – each sublist is a robot's task sequence
        title: Title of the plot
        save_path: Path to save the image. If None, shows the plot interactively.
    """
    plt.figure(figsize=(8, 6))
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta']

    # Plot all tasks
    for tid, (x, y) in planner_interface.TASK_COORDINATES.items():
        plt.scatter(x, y, c='black', s=40)
        plt.text(x + 0.01, y + 0.01, f'{tid}', fontsize=9)

    # Plot depot and path for each robot
    for r, task_seq in enumerate(individual):
        color = colors[r % len(colors)]
        depot = planner_interface.ROBOT_DEPOTS[r]

        # Build full path from depot → tasks
        # task_seq = [tid for tid in task_seq if tid >= 3]
        coords = [depot] + [planner_interface.TASK_COORDINATES[tid] for tid in task_seq] + [depot]


        xs, ys = zip(*coords)
        plt.plot(xs, ys, color=color, linewidth=2, label=f'Robot {r+1}')
        plt.scatter(xs[1:], ys[1:], c=color, s=60)  # task points only

        # Plot depot
        plt.scatter(*depot, c='black', marker='s', s=100)
        plt.text(depot[0] + 0.01, depot[1] + 0.01, f'D{r+1}', fontsize=9)

        # Mark start (depot) and end (last task)
        plt.scatter(xs[0], ys[0], c=color, marker='>', s=100)
        plt.scatter(xs[-1], ys[-1], c=color, marker='X', s=100)

    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.legend()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to: {save_path}")
    else:
        plt.show()
