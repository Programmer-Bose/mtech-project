import csv
import ast
from drl.planner_interface import generate_task_coordinates  # make sure this import is correct based on your structure
from utils.visualize import plot_best_solution

def load_individual_from_csv(csv_path, individual_idx, num_robots):
    """
    Loads a single individual (row) from a CSV file and parses the task assignments.
    
    Args:
        csv_path: Path to the CSV file.
        individual_idx: Row index (0-based) of the individual.
        num_robots: Number of robots (i.e., number of robot columns).
    
    Returns:
        List[List[int]] – robot-wise task assignment.
    """
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)  # skip header

        for i, row in enumerate(reader):
            if i == individual_idx:
                task_splits = []
                for j in range(num_robots):
                    try:
                        robot_tasks = ast.literal_eval(row[2 + j].strip())  # Assuming f1, f2 are first two columns
                    except:
                        robot_tasks = []
                    task_splits.append(robot_tasks)
                return task_splits

    raise IndexError(f"Individual index {individual_idx} out of bounds in file {csv_path}")

def plot_from_saved_csv(csv_path, individual_index=0, num_robots=3, save_path=None):
    individual = load_individual_from_csv(csv_path, individual_index, num_robots)
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        for i, row in enumerate(reader):
            if i == individual_index:
                f1 = row[0]
                f2 = row[1]
                break
    title = f"Individual {individual_index+1} | f1: {f1}, f2: {f2}"
    plot_best_solution(individual, title=title, save_path=save_path)

if __name__ == "__main__":
    csv_path = "results/final_population_200_200_20250709_202336.csv"
    individual_index = 8  
    num_robots = 4
    save_path = None
    generate_task_coordinates(num_tasks=30)  # Ensure task coordinates are generated

    # plot_from_saved_csv(csv_path, individual_index, num_robots, save_path)

    while True:
        user_input = input("Enter individual index (or 'q' to quit): ")
        if user_input.lower() == 'q':
            break
        try:
            idx = int(user_input)
            plot_from_saved_csv(csv_path, idx, num_robots, save_path)
        except ValueError:
            print("Please enter a valid integer index or 'q' to quit.")
        except IndexError as e:
            print(e)
