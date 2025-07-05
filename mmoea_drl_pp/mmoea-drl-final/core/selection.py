def select_next_generation(population, objective_values, flattened_population,
                           front_ranks, cscd_scores, pop_size):
    """
    Selects the next generation from combined parent + offspring population.

    Args:
        population: list of individuals (list of robot-wise task sequences)
        objective_values: list of tuples (f1, f2)
        flattened_population: list of flattened individuals
        front_ranks: list of integers (lower is better)
        cscd_scores: list of floats (higher is better)
        pop_size: how many individuals to keep

    Returns:
        selected_population: list of individuals
        selected_objectives: list of (f1, f2)
    """
    combined = list(zip(population, objective_values, flattened_population, front_ranks, cscd_scores))

    # Sort by: front rank (ascending), CSCD score (descending)
    combined.sort(key=lambda x: (x[3], -x[4]))

    # Select top-N
    selected = combined[:pop_size]
    selected_population = [item[0] for item in selected]
    selected_objectives = [item[1] for item in selected]

    return selected_population, selected_objectives

# from core.selection import select_next_generation

# population, objective_values = select_next_generation(
#     combined_population,
#     combined_objectives,
#     combined_flattened,
#     combined_fronts,
#     combined_cscd,
#     pop_size
# )
