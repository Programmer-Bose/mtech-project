# nsga2.py
def dominates(a, b):
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))

def assign_fronts(objective_values):
    N = len(objective_values)
    fronts = [None] * N
    domination_counts = [0] * N
    dominated_sets = [[] for _ in range(N)]
    current_front = []

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if dominates(objective_values[i], objective_values[j]):
                dominated_sets[i].append(j)
            elif dominates(objective_values[j], objective_values[i]):
                domination_counts[i] += 1
        if domination_counts[i] == 0:
            fronts[i] = 0
            current_front.append(i)

    # print(dominated_sets)
    # print(domination_counts)

    front = 0
    while current_front:
        next_front = []
        for p in current_front:
            for q in dominated_sets[p]:
                domination_counts[q] -= 1
                if domination_counts[q] == 0:
                    fronts[q] = front + 1
                    next_front.append(q)
        front += 1
        current_front = next_front

    return fronts

# if __name__ == "__main__":
#     # Example objective values for 5 individuals (minimization)
#     objective_values = [
#         [10, 2],
#         [20, 1],
#         [5,0.5],
#         [1,0.2]
#     ]
#     fronts = assign_fronts(objective_values)
#     print("Objective values:")
#     for i, obj in enumerate(objective_values):
#         print(f"Individual {i}: {obj}")
#     print("\nAssigned fronts:", fronts)

