import numpy as np
from core.encoding import unflatten_individual, flatten_individual
import random
from drl.planner_interface import TASK_COORDINATES

def euclidean_distance(a, b):
    max_len = max(len(a), len(b))
    a = a + [0] * (max_len - len(a))
    b = b + [0] * (max_len - len(b))
    return np.linalg.norm(np.array(a) - np.array(b))

def select_exemplar(index, flattened_pop, front_ranks, cscd_scores, F=0.3):
    current_rank = front_ranks[index]
    current_vector = flattened_pop[index]

    # Step 1: Find candidates from better fronts
    candidates = [i for i, r in enumerate(front_ranks) if r < current_rank]

    if not candidates:
        return index  # fallback: no better fronts

    # Step 2: Among those, pick top-CSCD elites (top F%)
    cscd_vals = [cscd_scores[i] for i in candidates]
    threshold = np.percentile(cscd_vals, F*100)
    elites = [i for i in candidates if cscd_scores[i] >= threshold]

    if not elites:
        elites = candidates  # fallback: use all candidates

    # Step 3: Select the closest elite in decision space
    distances = [euclidean_distance(current_vector, flattened_pop[i]) for i in elites]
    exemplar_idx = elites[np.argmin(distances)]
    return exemplar_idx


# def generate_offspring(parent, exemplar, mutation_prob=0.5):
    """
    Args:
        parent, exemplar: both are list-of-lists task sequences (robot-wise)

    Returns:
        new_individual: mutated child
    """
    # num_robots = len(parent)
    # tasks = set(t for r in parent for t in r)
    
    # # Flatten exemplar & parent
    # ex_flat = [t for r in exemplar for t in r]
    # ex_flat = [t for t in ex_flat if t in tasks]  # ensure same task set

    # # Crossover: start from exemplar
    # new_seq = ex_flat.copy()

    # # Mutation: randomly swap some tasks
    # if random.random() < mutation_prob:
    #     for _ in range(8): #need a hyperparameter
    #         i, j = random.sample(range(len(new_seq)), 2)
    #         new_seq[i], new_seq[j] = new_seq[j], new_seq[i]

    # # Redistribute to robots (uneven split)
    # splits = [0] * num_robots
    # for _ in new_seq:
    #     splits[random.randint(0, num_robots - 1)] += 1

    # robot_seq = []
    # idx = 0
    # for s in splits:
    #     robot_seq.append(new_seq[idx:idx+s])
    #     idx += s

    # return robot_seq


def de_generate_offspring_allocation(x_i, x_r1, x_r2, x_r3, F=0.5, CR=0.7, num_robots=3):
    x_i = np.array(x_i)
    x_r1 = np.array(x_r1)
    x_r2 = np.array(x_r2)
    x_r3 = np.array(x_r3)

    # Mutation in robot ID space (values between 0 and num_robots - 1)
    diff = x_r2 - x_r3
    mutant = x_r1 + F * diff

    # Add small noise to reduce bias, then floor and clamp
    mutant += np.random.uniform(-0.3, 0.3, size=mutant.shape)
    mutant = np.floor(mutant).astype(int)
    mutant = np.clip(mutant, 0, num_robots - 1)

    u_i = []
    j_rand = random.randint(0, len(x_i) - 1)

    for j in range(len(x_i)):
        r = random.random()
        if r < CR or j == j_rand:
            u_i.append(mutant[j])
        else:
            u_i.append(x_i[j])

    return u_i

def allocation_vector_to_robot_tasks(allocation_vector, num_robots):
    task_allocation = [[] for _ in range(num_robots)]
    for task_id, robot_id in enumerate(allocation_vector):
        robot_id = min(max(int(robot_id), 0), num_robots - 1)  # Ensure valid index
        task_allocation[robot_id].append(task_id)
    return task_allocation

def flatten(individual):
    return [task for robot in individual for task in robot]

def generate_offspring(parent, exemplar, mutation_prob=0.5, use_de=True, r1=None, r2=None, r3=None, F=0.5, CR=0.7):
    """
    Args:
        parent, exemplar: both are list-of-lists task sequences (robot-wise)
        use_de: whether to apply DE operator
        r1, r2, r3: optional DE parents (in list-of-lists form)

    Returns:
        new_individual: mutated child
    """
    num_robots = len(parent)
    tasks = set(t for r in parent for t in r)

    if use_de and r1 and r2 and r3:
        # Convert to allocation vector representation
        def to_allocation_vector(ind):
            vec = [0] * sum(len(r) for r in ind)
            for r_id, task_list in enumerate(ind):
                for t in task_list:
                    vec[t] = r_id
            return vec

        x_i = to_allocation_vector(parent)
        x_r1 = to_allocation_vector(r1)
        x_r2 = to_allocation_vector(r2)
        x_r3 = to_allocation_vector(r3)

        allocation_vec = de_generate_offspring_allocation(x_i, x_r1, x_r2, x_r3, F=F, CR=CR, num_robots=num_robots)
        allocation_vec = [min(max(int(rid), 0), num_robots - 1) for rid in allocation_vec]
        return allocation_vector_to_robot_tasks(allocation_vec, num_robots)

    # Otherwise use original exemplar-based mutation
    ex_flat = [t for r in exemplar for t in r if t in tasks]
    new_seq = ex_flat.copy()

    if random.random() < mutation_prob:
        for _ in range(3):
            i, j = random.sample(range(len(new_seq)), 2)
            new_seq[i], new_seq[j] = new_seq[j], new_seq[i]

    splits = [0] * num_robots
    for _ in new_seq:
        splits[random.randint(0, num_robots - 1)] += 1

    robot_seq, idx = [], 0
    for s in splits:
        robot_seq.append(new_seq[idx:idx + s])
        idx += s

    return robot_seq

def dbesm_selection(population, flattened_pop, front_ranks, cscd_scores):
    """
    Returns: new population (offspring) using DBESM logic
    """
    new_pop = []

    for i in range(len(population)):
        exemplar_idx = select_exemplar(i, flattened_pop, front_ranks, cscd_scores)
        exemplar = unflatten_individual(flattened_pop[exemplar_idx])
        # offspring = generate_offspring(population[i], exemplar)
        #choose r2 and r3 2 individuals from population randomly
        r2_idx, r3_idx = random.sample([j for j in range(len(population)) if j != i], 2)
        offspring = generate_offspring(population[i], exemplar, mutation_prob=0.5, use_de=True, r1=exemplar, r2=population[r2_idx], r3=population[r3_idx], F=0.7)
        new_pop.append(offspring)

    return new_pop


# if __name__ == "__main__":
#     from encoding import generate_individual, flatten_individual, unflatten_individual
#     from cscd import compute_cscd_scores
#     from clustering import cluster_population
#     from evaluation import evaluate_individual
#     from nsga2 import assign_fronts
#     import random

#     pop_size = 10
#     num_tasks = 10
#     num_robots = 3
#     num_clusters = 3

#     population = [generate_individual(num_tasks, num_robots) for _ in range(pop_size)]
#     flattened = [flatten_individual(ind) for ind in population]

#     def fake_drl(r, t): return len(t) * 10
#     objectives = [evaluate_individual(ind, fake_drl) for ind in population]
#     clusters = cluster_population(flattened, num_clusters)
#     fronts = assign_fronts(objectives)
#     cscd_scores = compute_cscd_scores(flattened, objectives, clusters, fronts)

#     offspring = dbesm_selection(population, flattened, fronts, cscd_scores)

#     for i, child in enumerate(offspring):
#         print(f"Child {i}: {child}")
