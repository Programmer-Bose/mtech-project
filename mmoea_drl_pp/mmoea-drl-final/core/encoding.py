import random
import numpy as np
from random import shuffle
# from drl.planner_interface import TASK_COORDINATES

def generate_individual(num_tasks, num_robots):
    tasks = list(range(num_tasks))
    random.shuffle(tasks)

    # Evenly distribute tasks (some robots may have 1 more/less)
    splits = np.array_split(tasks, num_robots)
    return [list(s) for s in splits]

    # tasks = list(range(num_tasks))
    # random.shuffle(tasks)

    # # Generate random split sizes that sum to num_tasks
    # split_sizes = [0] * num_robots
    # for _ in range(num_tasks):
    #     split_sizes[random.randint(0, num_robots - 1)] += 1

    # # Distribute tasks accordingly
    # individual = []
    # index = 0
    # for size in split_sizes:
    #     individual.append(tasks[index:index+size])
    #     index += size
    # # Ensure all tasks are assigned
    # assert sum(split_sizes) == num_tasks, "Not all tasks assigned!"
    # # print(individual)
    # return individual

def flatten_individual(individual):
    """Flatten robot-wise task list with -1 as separator"""
    flat = []
    for robot_tasks in individual:
        flat += robot_tasks + [-1]
    return flat[:-1]  # remove last separator

def remove_duplicates(population, objectives=None):
    seen = set()
    unique_pop = []
    unique_objs = [] if objectives else None

    for i, ind in enumerate(population):
        flat = tuple(flatten_individual(ind))  # Use your structure-preserving flattening
        if flat not in seen:
            seen.add(flat)
            unique_pop.append(ind)
            if objectives:
                unique_objs.append(objectives[i])

    if objectives:
        return unique_pop, unique_objs
    return unique_pop


def unflatten_individual(flattened):
    """Convert flattened form back to list-of-lists"""
    robots = []
    temp = []
    for t in flattened:
        if t == -1:
            robots.append(temp)
            temp = []
        else:
            temp.append(t)
    if temp:
        robots.append(temp)
    return robots


# if __name__ == "__main__":
#     num_tasks = 20
#     num_robots = 6

#     individual = generate_individual(num_tasks, num_robots)
#     print("Generated Individual:", individual)

#     flat = flatten_individual(individual)
#     print("Flattened:", flat)

#     restored = unflatten_individual(flat)
#     print("Restored:", restored)

#     assert sorted([t for r in individual for t in r]) == list(range(num_tasks)), "Missing tasks"
#     assert individual == restored, "Unflatten failed!"