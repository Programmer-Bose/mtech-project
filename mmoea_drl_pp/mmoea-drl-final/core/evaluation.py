# from drl.planner_interface import evaluate_drl
import random

def evaluate_individual(individual, drl_planner):
    total_path_length = 0.0
    max_robot_time = 0.0
    updated_individual = []

    for robot_id, task_seq in enumerate(individual):
        path_length, reordered_seq = drl_planner(robot_id, task_seq)
        total_path_length += path_length
        max_robot_time = max(max_robot_time, path_length)
        updated_individual.append(reordered_seq)

    # Replace individual's sequence with reordered one**
    individual[:] = updated_individual

    return total_path_length, max_robot_time


# def fake_drl_planner(robot_id, task_seq):
#     # Dummy logic: each task adds 10 units of path
#     return len(task_seq) * 10, random.shuffle(task_seq)

# if __name__ == "__main__":
#     from encoding import generate_individual

#     ind = generate_individual(num_tasks=20, num_robots=6)
#     print("Individual:", ind)

#     f1, f2 = evaluate_individual(ind, fake_drl_planner)
#     print("Total Path Length (f1):", f1)
#     print("Time Taken (f2):", f2)

