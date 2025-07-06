import time
from core.encoding import generate_individual, flatten_individual,remove_duplicates
from core.evaluation import evaluate_individual
from core.clustering import cluster_population
from core.cscd import compute_cscd_scores
from core.dbesm import dbesm_selection
from core.nsga2 import assign_fronts
from drl.planner_interface import generate_task_coordinates, init_drl_model, evaluate_drl,evaluate_drl_lns,TASK_COORDINATES
from utils.plot import plot_pareto_front
from utils.visualize import plot_best_solution
import random
from tqdm import tqdm
import time


sP = 1  # Scale factor for DBESM offspring generation

def initialize_population(pop_size, num_tasks, num_robots):
    return [generate_individual(num_tasks, num_robots) for _ in range(pop_size)]

def evaluate_population(population, drl_planner):
    return [evaluate_individual(ind, drl_planner) for ind in population]

def flatten_population(population):
    return [flatten_individual(ind) for ind in population]

def select_next_generation(pop, obj_vals, flat, fronts, cscd, N):
    """
    Select top-N individuals based on front + CSCD
    """
    combined = list(zip(pop, obj_vals, flat, fronts, cscd))
    combined.sort(key=lambda x: (x[3], -x[4]))  # sort by front asc, CSCD desc
    selected = combined[:N]
    return [x[0] for x in selected], [x[1] for x in selected]

def run_evolution(
    num_tasks=20,
    num_robots=3,
    pop_size=20,
    generations=40,
    num_clusters=5,
    drl_planner=evaluate_drl,
    drl_last_gen=evaluate_drl_lns
):
    

    # Step 1: Initialize
    population = initialize_population(pop_size, num_tasks, num_robots)
    # print(population[0])  # Print first individual for debugging

    for gen in tqdm(range(generations), desc="Evolution Progress"):
        # print(f"\n--- Generation {gen} ---")

        # Step 2: Flatten
        flattened = flatten_population(population)

        # Step 3: Evaluate
        if gen < generations - 1:
            # Use DRL planner for all generations except the last
            objective_values = evaluate_population(population, drl_planner)
        else:
            objective_values = evaluate_population(population, drl_last_gen)

        # Step 4: Assign fronts
        fronts = assign_fronts(objective_values)

        # Step 5: Clustering
        clusters = cluster_population(flattened, num_clusters)

        # Step 6: CSCD
        cscd_scores = compute_cscd_scores(flattened, objective_values, clusters, fronts)

        # Step 7: DBESM offspring generation
        offspring = dbesm_selection(sP * population, flattened, fronts, cscd_scores)

        # Combine populations
        combined_population = population + offspring
        # combined_flattened = flatten_population(combined_population)
        if gen < generations - 1:
            # Use DRL planner for all generations except the last
            combined_objectives = evaluate_population(combined_population, drl_planner)
        else:
            combined_objectives = evaluate_population(combined_population, drl_last_gen)

        combined_population, combined_objectives = remove_duplicates(combined_population, combined_objectives)
        combined_flattened = flatten_population(combined_population)

        # Recalculate fronts, clusters, CSCD for selection
        combined_fronts = assign_fronts(combined_objectives)
        combined_clusters = cluster_population(combined_flattened, num_clusters)
        combined_cscd = compute_cscd_scores(combined_flattened, combined_objectives, combined_clusters, combined_fronts)

        # Step 8: Select next generation
        population, objective_values = select_next_generation(
            combined_population,
            combined_objectives,
            combined_flattened,
            combined_fronts,
            combined_cscd,
            int(pop_size)
        )

        # Optional: Print best
        best = min(objective_values, key=lambda x: (x[0], x[1]))
        # tqdm.write(f"Best in gen {gen}: f1 = {best[0]}, f2 = {best[1]}")
        # print(f"Best in gen {gen}: f1 = {best[0]}, f2 = {best[1]}")
    
    return population, objective_values


if __name__ == "__main__":
    NUM_TASKS = 20
    # Generate new task coordinates each run
    generate_task_coordinates(NUM_TASKS)
    # Load DRL model once
    init_drl_model('tsp_ac_256_1L.pth', device='cuda')

    final_pop, final_objs = run_evolution(
        num_tasks=NUM_TASKS,
        num_robots=3,
        pop_size=200,
        generations=500,
        num_clusters=4,
        drl_planner=evaluate_drl,
        drl_last_gen=evaluate_drl_lns
    )

    # best_idx = final_objs.index(min(final_objs, key=lambda x: (x[0], x[1])))
    # best_solution = final_pop[best_idx]
    # print(f"\nBest Solution (f1, f2): {final_objs[best_idx]} at index {best_idx}")
    # print(f"Best Solution (Decision Space): {best_solution}")

    
    # plot_pareto_front(final_objs, title="Final Pareto Front", save_path="results/pareto_front.png")
    # plot_best_solution(best_solution, title="Best Multi-Robot Path", save_path="results/best_solution.png")

    # Step 1: Get top 5
    sorted_indices = sorted(range(len(final_objs)), key=lambda i: (final_objs[i][0], final_objs[i][1]))
    top_indices = sorted_indices[:10]
    top_solutions = [final_pop[i] for i in top_indices]
    top_objectives = [final_objs[i] for i in top_indices]

    # # Step 2: Plot top 10 paths
    for idx, (sol, obj) in enumerate(zip(top_solutions, top_objectives)):
        title = f"Top {idx+1} Path - f1={obj[0]:.2f}, f2={obj[1]:.2f}"
        save_path = f"results/path_sol_{time.strftime('%Y%m%d_%H%M%S')}_top_{idx+1}.png"
        plot_best_solution(sol, title=title, save_path=save_path)

    # Step 3: Pareto plot with highlights
    plot_pareto_front(top_objectives, title="Final Pareto Front", save_path=f"results/pareto_{time.strftime('%Y%m%d_%H%M%S')}.png")

    

    # print("\nFinal Pareto Front (Objective Space):")
    # for i, obj in enumerate(final_objs):
    #     print(f"Solution {i}: f1 = {obj[0]}, f2 = {obj[1]}")

    # print("\nFinal Pareto Set (Decision Space):")
    # for i, individual in enumerate(final_pop):
    #     print(f"Solution {i}:")
    #     for r, tasks in enumerate(individual):
    #         print(f"  Robot {r+1}: {tasks}")

