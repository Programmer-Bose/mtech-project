# from drl.planner_interface import evaluate_drl

def evaluate_individual(individual, drl_planner):
    """
    Args:
        individual: list of task sequences (one per robot)
        drl_planner: function(robot_id, task_seq) -> path_length

    Returns:
        f1 = total path length (sum of all robot paths)
        f2 = max path length (time taken to complete all tasks)
    """
    path_lengths = []

    for robot_id, task_seq in enumerate(individual):
        length = drl_planner(robot_id, task_seq)
        path_lengths.append(length)

    f1 = sum(path_lengths)
    f2 = max(path_lengths)
    return f1, f2

# def fake_drl_planner(robot_id, task_seq):
#     # Dummy logic: each task adds 10 units of path
#     return len(task_seq) * 10

# if __name__ == "__main__":
#     from encoding import generate_individual

#     ind = generate_individual(num_tasks=20, num_robots=6)
#     print("Individual:", ind)

#     f1, f2 = evaluate_individual(ind, fake_drl_planner)
#     print("Total Path Length (f1):", f1)
#     print("Time Taken (f2):", f2)