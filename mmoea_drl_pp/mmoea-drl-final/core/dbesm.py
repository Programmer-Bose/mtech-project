import numpy as np
from core.encoding import unflatten_individual, flatten_individual
import random
from drl.planner_interface import TASK_COORDINATES

def euclidean_distance(a, b):
    max_len = max(len(a), len(b))
    a = a + [0] * (max_len - len(a))
    b = b + [0] * (max_len - len(b))
    return np.linalg.norm(np.array(a) - np.array(b))

def select_exemplar(index, flattened_pop, front_ranks, cscd_scores, F=0.5):
    current_rank = front_ranks[index]
    current_vector = flattened_pop[index]

    # Step 1: Find candidates from better fronts
    candidates = [i for i, r in enumerate(front_ranks) if r < current_rank]

    if not candidates:
        return index  # fallback: no better fronts

    # Step 2: Among those, pick top-CSCD elites (top 50%)
    cscd_vals = [cscd_scores[i] for i in candidates]
    threshold = np.percentile(cscd_vals, 50)
    elites = [i for i in candidates if cscd_scores[i] >= threshold]

    if not elites:
        elites = candidates  # fallback: use all candidates

    # Step 3: Select the closest elite in decision space
    distances = [euclidean_distance(current_vector, flattened_pop[i]) for i in elites]
    exemplar_idx = elites[np.argmin(distances)]
    return exemplar_idx


def generate_offspring(parent, exemplar, mutation_prob=0.5):
    """
    Args:
        parent, exemplar: both are list-of-lists task sequences (robot-wise)

    Returns:
        new_individual: mutated child
    """
    num_robots = len(parent)
    tasks = set(t for r in parent for t in r)
    
    # Flatten exemplar & parent
    ex_flat = [t for r in exemplar for t in r]
    ex_flat = [t for t in ex_flat if t in tasks]  # ensure same task set

    # Crossover: start from exemplar
    new_seq = ex_flat.copy()

    # Mutation: randomly swap some tasks
    if random.random() < mutation_prob:
        for _ in range(3):
            i, j = random.sample(range(len(new_seq)), 2)
            new_seq[i], new_seq[j] = new_seq[j], new_seq[i]

    # Redistribute to robots (uneven split)
    splits = [0] * num_robots
    for _ in new_seq:
        splits[random.randint(0, num_robots - 1)] += 1

    robot_seq = []
    idx = 0
    for s in splits:
        robot_seq.append(new_seq[idx:idx+s])
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
        offspring = generate_offspring(population[i], exemplar)
        new_pop.append(offspring)

    return new_pop


# if __name__ == "__main__":
#     from encoding import generate_individual, flatten_individual
#     from cscd import compute_cscd_scores
#     from clustering import cluster_population
#     from evaluation import evaluate_individual
#     from nsga2 import assign_fronts

#     pop_size = 10
#     num_tasks = 9
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
