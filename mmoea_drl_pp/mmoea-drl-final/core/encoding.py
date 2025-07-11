import random
import numpy as np
from random import shuffle
# from drl.planner_interface import TASK_COORDINATES


def generate_individual(num_tasks, num_robots):
    tasks = np.random.permutation(num_tasks)  # random permutation of task indices
    # splits = np.array_split(tasks, num_robots)  # split roughly evenly
    # Split tasks unevenly among robots
    splits = []
    remaining = list(tasks)
    for i in range(num_robots - 1):
        # Ensure at least 1 task per robot, randomize split size
        max_split = len(remaining) - (num_robots - i - 1)
        split_size = random.randint(1, max_split)
        splits.append(remaining[:split_size])
        remaining = remaining[split_size:]
    splits.append(remaining)
    return [list(s) for s in splits]  # convert arrays to lists  

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
#     num_robots = 3

#     individual = generate_individual(num_tasks, num_robots)
#     print("Generated Individual:", individual)

#     flat = flatten_individual(individual)
#     print("Flattened:", flat)

#     restored = unflatten_individual(flat)
#     print("Restored:", restored)

#     population = [generate_individual(num_tasks, num_robots) for _ in range(2)]
#     print("Population:", population)

#     unique_population = remove_duplicates(population)
#     print("Unique Population:", unique_population)

#     if (len(population) == len(unique_population)):
#         print("No Duplicates!!")

    

#     assert sorted([t for r in individual for t in r]) == list(range(num_tasks)), "Missing tasks"
#     assert individual == restored, "Unflatten failed!"