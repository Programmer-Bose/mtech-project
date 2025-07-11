import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random
import math
from tqdm import tqdm # For LNS progress bar


ACTOR_MODEL = None
np.random.seed(42)

# ----- Actor (Unified Class) - IDENTICAL to your training setup -----
# This class is needed to load the state_dict from your .pth file.
class Actor(nn.Module):
    def __init__(self, embedding_dim=128, hidden_dim=128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.ff_layer = nn.Linear(2, embedding_dim)
        self.gru_encoder = nn.GRU(input_size=embedding_dim, hidden_size=embedding_dim, batch_first=True)
        self.gru_decoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.attention = self._Attention(hidden_dim)

    class _Attention(nn.Module):
        def __init__(self, hidden_dim):
            super().__init__()
            self.W1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.v = nn.Linear(hidden_dim, 1, bias=False)

        def forward(self, decoder_hidden, encoder_outputs, mask):
            decoder_hidden_expanded = decoder_hidden.transpose(0, 1) 
            score = self.v(torch.tanh(self.W1(encoder_outputs) + self.W2(decoder_hidden_expanded)))
            score = score.squeeze(-1) 
            score[mask == 0] = -1e9 
            attn_weights = F.softmax(score, dim=-1)
            return attn_weights

    def forward(self, input_coords, seq_len):
        embedded = self.ff_layer(input_coords)
        encoder_outputs, encoder_hidden = self.gru_encoder(embedded)

        batch_size = input_coords.size(0)
        
        decoder_input = embedded[:, 0, :].unsqueeze(1) 
        hidden = encoder_hidden 
        
        mask = torch.ones(batch_size, seq_len, device=input_coords.device)
        mask[:, 0] = 0 # Robot station (index 0) is visited

        tours_batch = [torch.zeros(batch_size, dtype=torch.long, device=input_coords.device)] 

        for _ in range(seq_len - 1): # Select seq_len - 1 other cities
            output, hidden = self.gru_decoder(decoder_input, hidden)
            attn_weights = self.attention(hidden, encoder_outputs, mask)
            
            selected = torch.argmax(attn_weights, dim=-1) # Argmax for deterministic inference
            
            tours_batch.append(selected)
            
            mask.scatter_(1, selected.unsqueeze(-1), 0)
            
            gather_indices = selected.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.embedding_dim)
            decoder_input = torch.gather(encoder_outputs, 1, gather_indices) 

        tours_batch_tensor = torch.stack(tours_batch, dim=1)
        
        return tours_batch_tensor, encoder_outputs
    

# ----- Functions for Loading and Inference from PTH model -----
def load_actor_model(model_path, embedding_dim=128, hidden_dim=128, device='cpu'):
    device = torch.device(device)
    actor = Actor(embedding_dim, hidden_dim).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    actor.load_state_dict(checkpoint['actor_state_dict'])
    
    actor.eval()
    print(f"Actor model loaded from {model_path} and set to evaluation mode on {device}.")
    return actor

def infer_tour_single_instance(actor_model, input_coords):
    device = next(actor_model.parameters()).device
    
    input_coords_batched = input_coords.unsqueeze(0).to(device)
    seq_len = input_coords.shape[0]

    with torch.no_grad():
        predicted_tours_batched, _ = actor_model(input_coords_batched, seq_len)
        predicted_tours = predicted_tours_batched.squeeze(0) # Remove batch dimension
        
        inferred_tour_indices = predicted_tours.cpu().numpy().tolist()
        
        original_coords_np = input_coords.cpu().numpy()
        inferred_tour_coords_np = original_coords_np[inferred_tour_indices]
        
        path_coords = np.vstack([inferred_tour_coords_np, inferred_tour_coords_np[0]])
        distances = np.linalg.norm(path_coords[1:] - path_coords[:-1], axis=1)
        tour_length = distances.sum()
        tour_reward = -tour_length

    return inferred_tour_indices, inferred_tour_coords_np, tour_reward

def infer_tour_batch(actor_model, batch_coords):
    """
    Infers tours for a batch of input coordinate sets using the loaded Actor model.

    Args:
        actor_model (Actor): The loaded Actor model in evaluation mode.
        batch_coords (torch.Tensor): Tensor of input coordinates for a batch.
                                        Shape: (batch_size, seq_len, 2)

    Returns:
        tuple:
            - inferred_tour_indices_list (list of lists): Each sublist is the sequence of visited node indices.
            - inferred_tour_coords_list (list of np.ndarray): Each array is (seq_len, 2) of inferred tour coordinates.
            - tour_lengths (list of float): Tour lengths for each instance in the batch.
    """
    device = next(actor_model.parameters()).device
    batch_coords = batch_coords.to(device)
    batch_size, seq_len, _ = batch_coords.shape

    with torch.no_grad():
        predicted_tours_batched, _ = actor_model(batch_coords, seq_len)  # (B, seq_len)
        inferred_tour_indices_list = []
        inferred_tour_coords_list = []
        tour_lengths = []

        for i in range(batch_size):
            predicted_tour = predicted_tours_batched[i].cpu().numpy().tolist()
            coords_np = batch_coords[i].cpu().numpy()
            inferred_coords = coords_np[predicted_tour]
            path_coords = np.vstack([inferred_coords, inferred_coords[0]])
            distances = np.linalg.norm(path_coords[1:] - path_coords[:-1], axis=1)
            tour_length = distances.sum()

            inferred_tour_indices_list.append(predicted_tour)
            inferred_tour_coords_list.append(inferred_coords)
            tour_lengths.append(tour_length)

    return inferred_tour_indices_list, inferred_tour_coords_list, tour_lengths

# ----- LNS Helper Functions -----

def calculate_tour_cost(tour_indices, coords):
    """Calculates the total length of a TSP tour."""
    if not tour_indices:
        return float('inf')
    
    path_coords = coords[tour_indices]
    
    # Add the return trip to the starting point (robot station)
    full_path = np.vstack([path_coords, coords[tour_indices[0]]])
    
    distances = np.linalg.norm(full_path[1:] - full_path[:-1], axis=1)
    return np.sum(distances)

def destroy_random(current_tour_indices, num_remove, seq_len):
    """
    Removes a random set of cities from the tour, excluding the robot station (index 0).
    Returns the partial tour indices and the removed cities' indices.
    """
    removable_indices_in_tour = [i for i, city_idx in enumerate(current_tour_indices) if city_idx != 0]
    
    num_remove = min(num_remove, len(removable_indices_in_tour))
    
    indices_to_remove_from_list = random.sample(removable_indices_in_tour, num_remove)
    
    removed_cities = [current_tour_indices[i] for i in sorted(indices_to_remove_from_list, reverse=True)]
    
    partial_tour_indices = list(current_tour_indices)
    for i in sorted(indices_to_remove_from_list, reverse=True):
        del partial_tour_indices[i]
        
    return partial_tour_indices, removed_cities

def repair_greedy(partial_tour_indices, removed_cities, coords):
    """
    Greedily re-inserts removed cities into the partial tour.
    """
    current_tour = list(partial_tour_indices)
    
    for city_to_insert in removed_cities:
        best_insertion_cost = float('inf')
        best_insertion_pos = -1
        
        for j in range(1, len(current_tour) + 1):
            temp_tour = list(current_tour)
            temp_tour.insert(j, city_to_insert)
            
            cost = calculate_tour_cost(temp_tour, coords)
            
            if cost < best_insertion_cost:
                best_insertion_cost = cost
                best_insertion_pos = j
        
        current_tour.insert(best_insertion_pos, city_to_insert)
        
    return current_tour

def simulated_annealing_acceptance(current_cost, new_cost, temperature):
    """
    Accepts new_cost if better, or with a probability if worse (Simulated Annealing).
    """
    if new_cost < current_cost:
        return True
    if temperature <= 0:
        return False
    
    probability = math.exp(-(new_cost - current_cost) / temperature)
    return random.random() < probability


# ----- Main LNS Function -----
def run_lns(initial_tour_indices, original_coords, seq_len, 
            max_iterations, num_remove_per_destroy, initial_temperature, cooling_rate):
    """
    Applies Large Neighborhood Search to improve an initial TSP tour.
    """
    current_tour_indices = list(initial_tour_indices)
    current_cost = calculate_tour_cost(current_tour_indices, original_coords)
    
    best_tour_indices = list(current_tour_indices)
    best_tour_cost = current_cost
    
    temperature = initial_temperature

    # print(f"\nStarting LNS. Initial Tour Cost: {current_cost:.4f}")

    # for iteration in tqdm(range(max_iterations), desc="LNS Progress"):
    for iteration in range(max_iterations):
        partial_tour_indices, removed_cities = destroy_random(current_tour_indices, num_remove_per_destroy, seq_len)
        
        new_tour_indices = repair_greedy(partial_tour_indices, removed_cities, original_coords)
        new_cost = calculate_tour_cost(new_tour_indices, original_coords)
        
        if simulated_annealing_acceptance(current_cost, new_cost, temperature):
            current_tour_indices = list(new_tour_indices)
            current_cost = new_cost
            
            if current_cost < best_tour_cost:
                best_tour_indices = list(current_tour_indices)
                best_tour_cost = current_cost

        temperature *= cooling_rate
        
    # print(f"LNS Finished. Best Tour Cost: {best_tour_cost:.4f}")
    return best_tour_indices, best_tour_cost


# ----- MODIFIED: Side-by-Side Plotting Function -----
def plot_comparison_tours(original_coords, 
                          rl_tour_coords, rl_tour_reward, 
                          lns_tour_coords, lns_tour_reward):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7)) # Two subplots side by side

    # --- Plot 1: RL Model Output ---
    # Plot all original cities as faint points
    ax1.scatter(original_coords[:, 0], original_coords[:, 1], color='lightgray', marker='o', s=50, zorder=1)
    
    # Plot the inferred tour path
    plot_path_coords_rl = np.vstack([rl_tour_coords, rl_tour_coords[0]])
    ax1.plot(plot_path_coords_rl[:, 0], plot_path_coords_rl[:, 1], 'b-o', linewidth=1.5, markersize=8, label='Tour Path', zorder=2)
    
    # Mark the visiting order and points
    for i, (x, y) in enumerate(rl_tour_coords):
        ax1.text(x + 0.01, y + 0.01, str(i + 1), fontsize=9, ha='center', va='center', color='black', weight='bold', zorder=3)
        ax1.scatter(x, y, color='blue', s=100, zorder=3)
    
    # Mark start and end points
    start_x_rl, start_y_rl = rl_tour_coords[0]
    ax1.plot(start_x_rl, start_y_rl, 'go', markersize=12, label='Start Point', zorder=4)
    ax1.text(start_x_rl + 0.02, start_y_rl + 0.02, 'S', fontsize=12, color='darkgreen', weight='bold', ha='center', va='center', zorder=5)
    
    end_x_rl, end_y_rl = rl_tour_coords[-1]
    ax1.plot(end_x_rl, end_y_rl, 'ro', markersize=12, label='End Point', zorder=4)
    ax1.text(end_x_rl - 0.02, end_y_rl - 0.02, 'E', fontsize=12, color='darkred', weight='bold', ha='center', va='center', zorder=5)
    
    ax1.set_title(f'RL Model Output (Length: {-rl_tour_reward:.4f})')
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend()


    # --- Plot 2: LNS Optimized Output ---
    # Plot all original cities as faint points
    ax2.scatter(original_coords[:, 0], original_coords[:, 1], color='lightgray', marker='o', s=50, zorder=1)
    
    # Plot the inferred tour path
    plot_path_coords_lns = np.vstack([lns_tour_coords, lns_tour_coords[0]])
    ax2.plot(plot_path_coords_lns[:, 0], plot_path_coords_lns[:, 1], 'g-o', linewidth=1.5, markersize=8, label='Tour Path', zorder=2) # Green line for LNS
    
    # Mark the visiting order and points
    for i, (x, y) in enumerate(lns_tour_coords):
        ax2.text(x + 0.01, y + 0.01, str(i + 1), fontsize=9, ha='center', va='center', color='black', weight='bold', zorder=3)
        ax2.scatter(x, y, color='green', s=100, zorder=3) # Green markers for LNS
    
    # Mark start and end points
    start_x_lns, start_y_lns = lns_tour_coords[0]
    ax2.plot(start_x_lns, start_y_lns, 'go', markersize=12, label='Start Point', zorder=4)
    ax2.text(start_x_lns + 0.02, start_y_lns + 0.02, 'S', fontsize=12, color='darkgreen', weight='bold', ha='center', va='center', zorder=5)
    
    end_x_lns, end_y_lns = lns_tour_coords[-1]
    ax2.plot(end_x_lns, end_y_lns, 'ro', markersize=12, label='End Point', zorder=4)
    ax2.text(end_x_lns - 0.02, end_y_lns - 0.02, 'E', fontsize=12, color='darkred', weight='bold', ha='center', va='center', zorder=5)
    
    ax2.set_title(f'LNS Optimized (Length: {-lns_tour_reward:.4f})')
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Y Coordinate')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.show()

# Loaded once, used for all inference

def init_drl_model(model_path='tsp_ac_256_1L.pth', device='cuda'):
    global ACTOR_MODEL
    ACTOR_MODEL = load_actor_model(model_path, embedding_dim=256, hidden_dim=256, device=device)


def evaluate_drl(robot_id, task_seq):
    global TASK_COORDINATES, ACTOR_MODEL

    if ACTOR_MODEL is None:
        raise ValueError("Call init_drl_model() first to load the DRL model.")

    # Handle empty task list
    if not task_seq:
        return 0.0,[]  # No path, no length
    # print(f"Evaluating DRL for Robot {robot_id} with tasks: {task_seq}")

    # Add depot coordinate at start
    depot_coord = ROBOT_DEPOTS[robot_id]
    # task_seq = [tid for tid in task_seq if tid not in [0, 1, 2]]
    coords = [depot_coord] + [TASK_COORDINATES[tid] for tid in task_seq]
    

    coords_np = np.array(coords, dtype=np.float32)
    coords_tensor = torch.from_numpy(coords_np)

    tour_indices, _, reward = infer_tour_single_instance(ACTOR_MODEL, coords_tensor)
    path_length = -reward

    # Add distance from last task to robot depot to the path length
    # if len(tour_indices) > 1:
        # last_task_idx = tour_indices[-1]
        # last_task_coord = coords_np[last_task_idx]
        # depot_coord_np = np.array(depot_coord, dtype=np.float32)
        # extra_distance = np.linalg.norm(last_task_coord - depot_coord_np)
        # path_length += extra_distance

    # Convert DRL tour indices back to task IDs
    # tour_indices[0] is depot (0), so ignore it
    optimized_sequence = [task_seq[i - 1] for i in tour_indices if i != 0]

    return path_length, optimized_sequence

def evaluate_drl_lns(robot_id, task_seq):
    global TASK_COORDINATES, ACTOR_MODEL

    if ACTOR_MODEL is None:
        raise ValueError("Call init_drl_model() first to load the DRL model.")

    if not task_seq:
        return 0.0,[]  # No path, no travel

    # Add depot coordinate at start
    depot_coord = ROBOT_DEPOTS[robot_id]
    # task_seq = [tid for tid in task_seq if tid not in [0, 1, 2]]
    coords = [depot_coord] + [TASK_COORDINATES[tid] for tid in task_seq]

    coords_np = np.array(coords, dtype=np.float32)
    coords_tensor = torch.from_numpy(coords_np)

    # Step 1: Inference using DRL model
    rl_tour_indices, _, rl_tour_reward = infer_tour_single_instance(ACTOR_MODEL, coords_tensor)
    rl_tour_cost = -rl_tour_reward  # Negative reward is positive cost

    # Step 2: Refine using LNS
    seq_len = len(task_seq)
    lns_iterations = 100
    lns_removal = 5
    initial_temp = 1.0
    cooling = 0.995

    refined_tour_indices, refined_cost = run_lns(
        rl_tour_indices,
        coords_np,
        seq_len,
        lns_iterations,
        lns_removal,
        initial_temp,
        cooling
    )

    # Add distance from last task to robot depot to the path length
    # if len(refined_tour_indices) > 1:
    #     last_task_idx = refined_tour_indices[-1]
    #     last_task_coord = coords_np[last_task_idx]
    #     depot_coord_np = np.array(depot_coord, dtype=np.float32)
    #     extra_distance = np.linalg.norm(last_task_coord - depot_coord_np)
    #     refined_cost += extra_distance


    optimized_sequence = [task_seq[i - 1] for i in refined_tour_indices if i != 0]
    
    return refined_cost, optimized_sequence



TASK_COORDINATES = {}

ROBOT_DEPOTS = {
    0: (0.5, 0.05),   # Robot 0's base
    1: (0.5, 0.05),   # Robot 1's base
    2: (0.5, 0.95),   # Robot 2's base
    3: (0.5, 0.95),   # Robot 3's base
    4: (0.05, 0.5),   # Robot 2's base
    5: (0.05, 0.5),   # Robot 3's base
}


def generate_task_coordinates(num_tasks):
    """
    Generates random (x, y) coordinates for each task ID.
    Task 0 is the robot station and placed at a fixed point.

    Updates global TASK_COORDINATES.
    """
    global TASK_COORDINATES
    if num_tasks == 30:
        TASK_COORDINATES = {0: (0.69705, 0.65912), 1: (0.72966, 0.74986), 2: (0.1696, 0.0068), 
                            3: (0.84208, 0.32632), 4: (0.83636, 0.36766), 5: (0.89403, 0.81913), 
                            6: (0.10039, 0.20462), 7: (0.54908, 0.06478), 8: (0.13489, 0.9508), 
                            9: (0.06335, 0.23856), 10: (0.79052, 0.4825), 11: (0.37287, 0.58046), 
                            12: (0.78679, 0.00468), 13: (0.85591, 0.9104), 14: (0.03169, 0.23075), 
                            15: (0.16304, 0.64752), 16: (0.56675, 0.21116), 17: (0.09113, 0.81652), 
                            18: (0.52155, 0.74205), 19: (0.39718, 0.93534), 20: (0.12136, 0.68844), 
                            21: (0.60087, 0.20341), 22: (0.85873, 0.20238), 23: (0.56802, 0.30087), 
                            24: (0.34087, 0.24421), 25: (0.69079, 0.22627), 26: (0.77693, 0.21814), 
                            27: (0.23676, 0.75051), 28: (0.57872, 0.44212), 29: (0.8314, 0.28604)}
    elif num_tasks == 40:
        TASK_COORDINATES = {0: (0.28622, 0.05656), 1: (0.72577, 0.35559), 2: (0.71636, 0.50594), 3: (0.55564, 0.13739), 4: (0.00603, 0.61574), 
                            5: (0.56315, 0.18749), 6: (0.66332, 0.3703), 7: (0.21929, 0.23674), 8: (0.55631, 0.95716), 9: (0.78265, 0.8054), 
                            10: (0.72049, 0.08855), 11: (0.71608, 0.69803), 12: (0.45737, 0.5548), 13: (0.91572, 0.59762), 14: (0.01751, 0.65288), 
                            15: (0.07649, 0.42422), 16: (0.17067, 0.47288), 17: (0.02204, 0.80086), 18: (0.18427, 0.15977), 19: (0.11608, 0.36249), 
                            20: (0.64523, 0.93359), 21: (0.06693, 0.72292), 22: (0.70244, 0.70216), 23: (0.90778, 0.17913), 24: (0.04519, 0.67917), 
                            25: (0.09136, 0.14897), 26: (0.55026, 0.48733), 27: (0.60853, 0.23141), 28: (0.46426, 0.69563), 29: (0.99101, 0.05494), 
                            30: (0.18471, 0.96555), 31: (0.41737, 0.17116), 32: (0.3363, 0.64772), 33: (0.76816, 0.12956), 34: (0.73206, 0.89114), 
                            35: (0.06908, 0.3784), 36: (0.78401, 0.47397), 37: (0.80574, 0.03623), 38: (0.12299, 0.43387), 39: (0.67929, 0.61887)}
    elif num_tasks == 50:
        TASK_COORDINATES = {0: (0.39534, 0.68756), 1: (0.4424, 0.60378), 2: (0.20967, 0.29293), 3: (0.03024, 0.01028), 4: (0.76231, 0.06503), 
                            5: (0.4733, 0.53059), 6: (0.68235, 0.17453), 7: (0.90423, 0.36469), 8: (0.80707, 0.02214), 9: (0.00513, 0.58839), 
                            10: (0.59005, 0.45084), 11: (0.64651, 0.69564), 12: (0.80693, 0.04008), 13: (0.46085, 0.3875), 14: (0.43236, 0.71253), 
                            15: (0.77861, 0.06421), 16: (0.35728, 0.28466), 17: (0.77865, 0.99453), 18: (0.58099, 0.88816), 19: (0.85589, 0.43875), 
                            20: (0.25906, 0.69002), 21: (0.23123, 0.99426), 22: (0.04942, 0.44741), 23: (0.78031, 0.81734), 24: (0.97362, 0.14262), 
                            25: (0.73617, 0.86862), 26: (0.66363, 0.29071), 27: (0.04696, 0.4755), 28: (0.10449, 0.33187), 29: (0.79317, 0.98257), 
                            30: (0.56858, 0.46599), 31: (0.57042, 0.99747), 32: (0.52551, 0.1575), 33: (0.85998, 0.37287), 34: (0.79473, 0.23393), 
                            35: (0.96237, 0.29968), 36: (0.94451, 0.56889), 37: (0.68185, 0.14454), 38: (0.67245, 0.48153), 39: (0.38454, 0.74664), 
                            40: (0.10245, 0.08172), 41: (0.44812, 0.37467), 42: (0.57389, 0.088), 43: (0.00088, 0.79179), 44: (0.89126, 0.10573), 
                            45: (0.14148, 0.27892), 46: (0.76024, 0.51157), 47: (0.61734, 0.4529), 48: (0.17526, 0.17778), 49: (0.33901, 0.57512)}
    print(f"Generated TASK_COORDINATES: {TASK_COORDINATES}")


# --- Main Execution Block ---
if __name__ == '__main__':
    # Hyperparameters (must match training and ONNX export)
    embedding_dim = 256
    hidden_dim = 256
    seq_len = 20 # Number of cities (including robot station at index 0)

    # Path to your trained PyTorch model
    pth_model_path = 'tsp_ac_256_1L.pth' 

    # LNS Parameters
    lns_max_iterations = 1000 # Number of destroy-repair cycles
    lns_num_remove = 5        # Number of cities to remove in each destroy step (excluding station)
    lns_initial_temp = 1.0    # Starting temperature for SA
    lns_cooling_rate = 0.999  # Rate at which temperature decreases (e.g., 0.999 for slow cooling)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    try:
        # 1. Load the Actor model (from .pth file)
        loaded_actor = load_actor_model(pth_model_path, embedding_dim, hidden_dim, device)

        # 2. Prepare a single new TSP instance with fixed robot station
        # Robot station is at [0.5, 0.5]
        robot_station_coord = np.array([[0.55, 0.85]], dtype=np.float32)
        # np.random.seed(42) # For reproducibility of this specific problem instance
        # random_other_coords = np.random.rand(seq_len - 1, 2).astype(np.float32)

        #----------------------------
        # Generate random coordinates from a non-uniform distribution (e.g., normal distribution centered at 0.5, 0.5)
        random_other_coords = np.clip(
            np.random.normal(loc=0.5, scale=0.25, size=(seq_len - 1, 2)).astype(np.float32),
            0.0, 1.0
        )
        #----------------------------

        single_instance_coords = np.concatenate([robot_station_coord, random_other_coords], axis=0)
        
        # Convert to torch tensor for inference
        single_instance_coords_torch = torch.from_numpy(single_instance_coords).float()
        
        print(f"\n--- Initial Tour from RL Model ---")
        # 3. Infer initial tour using the RL model
        rl_tour_indices, rl_tour_coords_np, rl_tour_reward = \
            infer_tour_single_instance(loaded_actor, single_instance_coords_torch)
        
        print(f"RL Inferred tour (indices): {rl_tour_indices}")
        print(f"RL Tour Length: {-rl_tour_reward:.4f}")

        # 4. Apply Large Neighborhood Search
        print(f"\n--- Applying LNS to improve RL tour ---")
        lns_best_tour_indices, lns_best_tour_cost = \
            run_lns(rl_tour_indices, single_instance_coords, seq_len,
                    lns_max_iterations, lns_num_remove, lns_initial_temp, lns_cooling_rate)
        
        lns_best_tour_coords_np = single_instance_coords[lns_best_tour_indices]
        lns_best_tour_reward = -lns_best_tour_cost

        print(f"LNS Optimized tour (indices): {lns_best_tour_indices}")
        print(f"LNS Optimized Tour Length: {-lns_best_tour_reward:.4f}")
        
        # 5. Plot both tours side-by-side for comparison
        plot_comparison_tours(single_instance_coords,
                              rl_tour_coords_np, rl_tour_reward,
                              lns_best_tour_coords_np, lns_best_tour_reward)

        print(f"\nOptimization Summary:")
        print(f"  RL Model Tour Length: {-rl_tour_reward:.4f}")
        print(f"  LNS Optimized Tour Length: {-lns_best_tour_reward:.4f}")
        if abs(lns_best_tour_cost) < abs(rl_tour_reward):
             print(f"  LNS improved tour by: {(-rl_tour_reward) - (-lns_best_tour_reward):.4f}")
        else:
             print("  LNS did not find a better solution (or found one of similar quality).")

    except FileNotFoundError:
        print(f"Error: Model file not found at {pth_model_path}.")
        print("Please ensure you have run the training script and the model file exists.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

