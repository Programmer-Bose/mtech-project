import numpy as np
from collections import defaultdict

def normalize_vectors(vectors):
    """Normalize vectors to [0, 1] range dimension-wise."""
    array = np.array(vectors, dtype=np.float32)
    min_vals = np.min(array, axis=0)
    max_vals = np.max(array, axis=0)
    denom = np.where(max_vals - min_vals == 0, 1, max_vals - min_vals)
    return (array - min_vals) / denom

# Compute crowding distance in decision space for each cluster
def compute_decision_space_crowding(flattened_population, cluster_labels):
    cluster_map = defaultdict(list)
    for i, label in enumerate(cluster_labels):
        cluster_map[label].append(i)

    crowding_x = np.zeros(len(flattened_population))

    for cluster_id, indices in cluster_map.items():
        if len(indices) <= 2:
            # Boundary or singleton cluster → assign high crowding
            for idx in indices:
                crowding_x[idx] = float('inf')
            continue

        cluster_vectors = [flattened_population[i] for i in indices]
        max_len = max(len(v) for v in cluster_vectors)
        padded = [v + [0] * (max_len - len(v)) for v in cluster_vectors]
        norm_vectors = normalize_vectors(padded)

        # Calculate crowding for each dimension
        cluster_crowding = np.zeros(len(indices))
        for d in range(norm_vectors.shape[1]):
            sorted_idx = np.argsort(norm_vectors[:, d])
            cluster_crowding[sorted_idx[0]] = float('inf')
            cluster_crowding[sorted_idx[-1]] = float('inf')
            for j in range(1, len(indices) - 1):
                prev_val = norm_vectors[sorted_idx[j - 1], d]
                next_val = norm_vectors[sorted_idx[j + 1], d]
                cluster_crowding[sorted_idx[j]] += (next_val - prev_val)

        for i, idx in enumerate(indices):
            crowding_x[idx] = cluster_crowding[i]

    return crowding_x.tolist()

# Compute crowding distance in objective space for each front
def compute_objective_space_crowding(objective_values, front_labels):
    front_map = defaultdict(list)
    for i, front in enumerate(front_labels):
        front_map[front].append(i)

    crowding_f = np.zeros(len(objective_values))

    for front_id, indices in front_map.items():
        if len(indices) <= 2:
            for idx in indices:
                crowding_f[idx] = float('inf')
            continue

        front_objs = [objective_values[i] for i in indices]
        norm_objs = normalize_vectors(front_objs)

        front_crowding = np.zeros(len(indices))
        for d in range(norm_objs.shape[1]):
            sorted_idx = np.argsort(norm_objs[:, d])
            front_crowding[sorted_idx[0]] = float('inf')
            front_crowding[sorted_idx[-1]] = float('inf')
            for j in range(1, len(indices) - 1):
                prev_val = norm_objs[sorted_idx[j - 1], d]
                next_val = norm_objs[sorted_idx[j + 1], d]
                front_crowding[sorted_idx[j]] += (next_val - prev_val)

        for i, idx in enumerate(indices):
            crowding_f[idx] = front_crowding[i]

    return crowding_f.tolist()


def compute_cscd(crowding_x, crowding_f):
    # Remove inf before computing mean
    finite_cx = [v for v in crowding_x if not np.isinf(v) and not np.isnan(v)]
    finite_cf = [v for v in crowding_f if not np.isinf(v) and not np.isnan(v)]

    # Safe fallback if empty
    finite_cx = [v for v in crowding_x if np.isfinite(v)]
    avg_x = np.mean(finite_cx) if finite_cx else 0.0
    avg_f = np.mean(finite_cf) if finite_cf else 0.0

    cscd = []
    for cx, cf in zip(crowding_x, crowding_f):
        # Replace nan/inf with safe defaults
        cx = cx if np.isfinite(cx) else 0.0
        cf = cf if np.isfinite(cf) else 0.0

        if cx > avg_x or cf > avg_f:
            cscd.append(max(cx, cf))
        else:
            cscd.append(min(cx, cf))
    return cscd



def compute_cscd_scores(flattened_population, objective_values, cluster_labels, front_labels):
    crowding_x = compute_decision_space_crowding(flattened_population, cluster_labels)
    crowding_f = compute_objective_space_crowding(objective_values, front_labels)
    cscd = compute_cscd(crowding_x, crowding_f)
    return cscd

# if __name__ == "__main__":
#     from encoding import generate_individual, flatten_individual

#     pop_size = 10
#     num_tasks = 20
#     num_robots = 6

#     population = [generate_individual(num_tasks, num_robots) for _ in range(pop_size)]
#     flat = [flatten_individual(ind) for ind in population]
#     objectives = [(i, 100 - i) for i in range(pop_size)]  # Dummy values
#     clusters = [i % 3 for i in range(pop_size)]  # Dummy clusters
#     fronts = [i // 3 for i in range(pop_size)]   # Dummy fronts

#     cscd_scores = compute_cscd_scores(flat, objectives, clusters, fronts)
#     for i, score in enumerate(cscd_scores):
#         print(f"Ind {i}: CSCD = {score:.3f}")
