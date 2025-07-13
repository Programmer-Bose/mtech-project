import time
from core.encoding import generate_individual, flatten_individual,remove_duplicates
from core.evaluation import evaluate_individual
from core.clustering import cluster_population
from core.cscd import compute_cscd_scores
from core.dbesm import dbesm_selection
from core.nsga2 import assign_fronts
from drl.planner_interface import generate_task_coordinates, init_drl_model, evaluate_drl,evaluate_drl_lns,TASK_COORDINATES
from utils.plot import plot_pareto_front
from utils.visualize import plot_best_solution, plot_pareto_gen
import random
from tqdm import tqdm
import pandas as pd
import ast
import os
import numpy as np




# def initialize_population(pop_size, num_tasks, num_robots):
#     return [generate_individual(num_tasks, num_robots) for _ in range(pop_size)]


def initialize_population(pop_size, num_tasks, num_robots, resume_file=None):
    if resume_file and os.path.isfile(resume_file):
        print(f"🔄 Resuming from saved population: {resume_file}")
        df = pd.read_csv(resume_file)

        population = []
        for _, row in df.iterrows():
            individual = []
            for i in range(num_robots):
                robot_col = f"robot_{i+1}"
                task_list = ast.literal_eval(row[robot_col])
                individual.append(task_list)
            population.append(individual)
        # Remove worst 20% individuals (based on f1, f2 from CSV)
        num_remove = int(0.5 * len(population))
        if num_remove > 0:
            # Get objectives directly from CSV columns
            loaded_objectives = [(row['f1'], row['f2']) for _, row in df.iterrows()]
            # Sort by objectives (assuming lower is better for both f1, f2)
            sorted_indices = sorted(range(len(population)), key=lambda i: (loaded_objectives[i][1], loaded_objectives[i][0]))
            # Remove worst
            keep_indices = sorted_indices[:-num_remove]
            population = [population[i] for i in keep_indices]

            # Add random 20% individuals
            num_add = num_remove
            random_individuals = [generate_individual(num_tasks, num_robots) for _ in range(num_add)]
            population.extend(random_individuals)
        return population
    else:
        print("🆕 Generating fresh random population")
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
    threshold = 2.3
    combined = [item for item in combined if abs(item[1][0] - item[1][1]) >= threshold]
    combined.sort(key=lambda x: (x[3], -x[4]))  # sort by front asc, CSCD desc
    selected = combined[:N]
    return [x[0] for x in selected], [x[1] for x in selected]

def next_gen_de(parent, p_obj, offspring, off_obj):
    combined_population = parent + offspring
    combined_objectives = p_obj + off_obj

    # Ensure objective values are scalar floats, not numpy arrays
    def to_scalar(val):
        if isinstance(val, np.ndarray):
            return float(val.item()) if val.size == 1 else float(val[0])
        return float(val)

    cleaned_objectives = [(to_scalar(obj[0]), to_scalar(obj[1])) for obj in combined_objectives]

    # Sort by f1, then f2
    sorted_indices = sorted(range(len(combined_population)),
                            key=lambda i: (cleaned_objectives[i][0], cleaned_objectives[i][1]))

    N = len(parent)
    next_gen = [combined_population[i] for i in sorted_indices[:N]]
    next_gen_objs = [cleaned_objectives[i] for i in sorted_indices[:N]]

    return next_gen, next_gen_objs

  
def run_evolution(
    num_tasks=20,
    num_robots=3,
    pop_size=20,
    generations=40,
    num_clusters=5,
    drl_planner=evaluate_drl,
    drl_last_gen=evaluate_drl_lns,
    resume_file=None
):
    

    # Step 1: Initialize
    population = initialize_population(pop_size, num_tasks, num_robots, resume_file=resume_file)
    best_in_gen = []  # Store best solutions for each generation

    for gen in range(generations):
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
        offspring = dbesm_selection(population, flattened, fronts, cscd_scores,num_robots)

    
        # Optionally, you can use next_gen and next_gen_objs for further processing
        # Combine populations
        # combined_population = population + offspring
        # combined_flattened = flatten_population(combined_population)

        # Only evaluate offspring objectives and combine with parent objectives
        if gen < generations - 1:
            offspring_objectives = evaluate_population(offspring, drl_planner)
        else:
            offspring_objectives = evaluate_population(offspring, drl_last_gen)
        
        
        # population,objective_values = next_gen_de(population,objective_values,offspring,offspring_objectives)

        combined_population = population + offspring
        combined_objectives = objective_values + offspring_objectives

        # if gen < generations - 1:
        #     # Use DRL planner for all generations except the last
        #     combined_objectives = evaluate_population(combined_population, drl_planner)
        # else:
        #     combined_objectives = evaluate_population(combined_population, drl_last_gen)

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
        print(f"Best in gen {gen}: f1 = {best[0]}, f2 = {best[1]}")
        # print(f"Combined Population Size:{len(combined_population)}")
        # print(f"Population Size:{len(population)}")
        #save best from each generation for plotting in a list
        best_in_gen.append(best)


    return population, objective_values, best_in_gen


if __name__ == "__main__":
    #Hyperparameters
    NUM_TASKS = 30
    NUM_ROBOTS = 4
    POP_SIZE = 200
    GENERATIONS = 100
    NUM_CLUSTERS = 6

    #print hyperparameter
    print(f"Hyperparameters:\n  NUM_TASKS = {NUM_TASKS}\n  NUM_ROBOTS = {NUM_ROBOTS}\n POP_SIZE = {POP_SIZE}\n  GENERATIONS = {GENERATIONS}\n  NUM_CLUSTERS = {NUM_CLUSTERS}")

    # Generate new task coordinates each run
    generate_task_coordinates(NUM_TASKS)
    # Load DRL model once
    init_drl_model('tsp_ac_256_1L.pth', device='cuda')

    final_pop, final_objs, best_in_gen = run_evolution(
        num_tasks=NUM_TASKS,
        num_robots=NUM_ROBOTS,
        pop_size=POP_SIZE,
        generations=GENERATIONS,
        num_clusters=NUM_CLUSTERS,
        drl_planner=evaluate_drl,
        drl_last_gen=evaluate_drl_lns,
        resume_file=None
        # resume_file="results/final_population_100_200_20250710_223519.csv"  # Set to a CSV file path to resume from saved population
    )

    
    best_idx = final_objs.index(min(final_objs, key=lambda x: (x[0], x[1])))
    best_solution = final_pop[best_idx]
    print(f"\nBest Solution (f1, f2): {final_objs[best_idx]} at index {best_idx}")
    print(f"Best Solution (Decision Space): {best_solution}")

    # Visualize results
    plot_pareto_front(final_objs, title="Final Pareto Front", save_path=f"results/pareto_front_{NUM_TASKS}T_{time.strftime('%Y%m%d_%H%M%S')}.png")
    plot_best_solution(best_solution, title=f"Best Path-{final_objs[best_idx]}", save_path=f"results/{NUM_TASKS}T_best_solution_{time.strftime('%Y%m%d_%H%M%S')}.png")
    # plot_pareto_gen(best_in_gen, title="Pareto Front Over Generations", save_path=f"results/pareto_gen_{time.strftime('%Y%m%d_%H%M%S')}.png")


    # Save final population and their objectives to CSV
    # Each row: f1, f2, robot_1, robot_2, ..., robot_N
    data = []
    for ind, obj in zip(final_pop, final_objs):
        row = {
            'f1': obj[0],
            'f2': obj[1]
        }
        # Each robot's task list as a string
        for i, robot_tasks in enumerate(ind):
            row[f'robot_{i+1}'] = str(robot_tasks)
        data.append(row)
    df = pd.DataFrame(data)
    csv_path = f"results/final_population_{GENERATIONS}_{POP_SIZE}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Final population saved to {csv_path}")
    
    

    # print("\nFinal Pareto Front (Objective Space):")
    # for i, obj in enumerate(final_objs):
    #     print(f"Solution {i}: f1 = {obj[0]}, f2 = {obj[1]}")

    # print("\nFinal Pareto Set (Decision Space):")
    # for i, individual in enumerate(final_pop):
    #     print(f"Solution {i}:")
    #     for r, tasks in enumerate(individual):
    #         print(f"  Robot {r+1}: {tasks}")

