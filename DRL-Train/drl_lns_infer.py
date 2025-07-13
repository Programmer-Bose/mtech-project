import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random
import math
from tqdm import tqdm # For LNS progress bar

# ----- Actor (Unified Class) - REPEATED FOR COMPLETENESS -----
# Make sure this class definition is IDENTICAL to the one used for training
# so that the loaded state_dict matches the model's structure.
class Actor(nn.Module):
    def __init__(self, embedding_dim=256, hidden_dim=256):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Encoder part
        self.ff_layer = nn.Linear(2, embedding_dim)
        self.gru_encoder = nn.GRU(input_size=embedding_dim, hidden_size=embedding_dim, batch_first=True)

        # Decoder part (which includes Attention)
        self.gru_decoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.attention = self._Attention(hidden_dim) # Using inner Attention class

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
            score[mask == 0] = -1e9 # Mask visited
            attn_weights = F.softmax(score, dim=-1)
            return attn_weights

    def forward(self, input_coords, seq_len):
        # input_coords: (B, S, 2)

        embedded = self.ff_layer(input_coords)
        encoder_outputs, encoder_hidden = self.gru_encoder(embedded)

        batch_size = input_coords.size(0)
        
        mask = torch.ones(batch_size, seq_len, device=input_coords.device)
        decoder_input = torch.zeros(batch_size, 1, self.embedding_dim, device=input_coords.device)
        hidden = encoder_hidden 
        
        tours_batch = [] 

        for _ in range(seq_len):
            output, hidden = self.gru_decoder(decoder_input, hidden)
            attn_weights = self.attention(hidden, encoder_outputs, mask)
            
            selected = torch.argmax(attn_weights, dim=-1) # (B,)
            
            tours_batch.append(selected)
            
            mask.scatter_(1, selected.unsqueeze(-1), 0)
            
            gather_indices = selected.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.embedding_dim)
            decoder_input = torch.gather(encoder_outputs, 1, gather_indices) # No .detach() needed in no_grad block

        tours_batch_tensor = torch.stack(tours_batch, dim=1) # (B, seq_len)
        
        return tours_batch_tensor, encoder_outputs 

# ----- Functions for Loading and Inference -----

def load_actor_model(model_path, embedding_dim=128, hidden_dim=128, device='cpu'):
    """
    Loads a trained Actor model from a checkpoint.

    Args:
        model_path (str): Path to the saved model checkpoint (.pth file).
        embedding_dim (int): Embedding dimension used during training.
        hidden_dim (int): Hidden dimension used during training.
        device (str or torch.device): Device to load the model onto ('cpu' or 'cuda').

    Returns:
        Actor: The loaded Actor model in evaluation mode.
    """
    device = torch.device(device)
    actor = Actor(embedding_dim, hidden_dim).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    actor.load_state_dict(checkpoint['actor_state_dict'])
    
    actor.eval() # Set the model to evaluation mode
    print(f"Actor model loaded from {model_path} and set to evaluation mode on {device}.")
    return actor

def infer_tour_single_instance(actor_model, input_coords):
    """
    Infers the optimal tour for a single given set of input coordinates
    using the loaded Actor model.

    Args:
        actor_model (Actor): The loaded Actor model in evaluation mode.
        input_coords (torch.Tensor): A tensor of input coordinates for a SINGLE instance.
                                      Shape: (Seq_len, 2)

    Returns:
        tuple:
            - inferred_tour_indices (list): The sequence of visited node indices (Python list).
            - inferred_tour_coords (np.ndarray): The corresponding coordinates of the inferred tour.
                                                  Shape: (Seq_len, 2)
            - tour_reward (float): The calculated reward (negative tour length) for the inferred tour.
    """
    device = next(actor_model.parameters()).device # Get model's device
    
    # Add a batch dimension for the Actor's forward pass (expects B, S, 2)
    input_coords_batched = input_coords.unsqueeze(0).to(device) # (1, Seq_len, 2)
    seq_len = input_coords.shape[0]

    with torch.no_grad(): # No gradient calculation needed for inference
        # The forward method returns tours_batch_tensor and encoder_outputs
        predicted_tours_batched, _ = actor_model(input_coords_batched, seq_len)
        
        # Remove the batch dimension
        predicted_tours = predicted_tours_batched.squeeze(0) # (Seq_len,)

        # Calculate reward for the inferred tour
        # tours_batch: (Seq_len,) - indices
        
        # Gather coordinates for the single tour
        # input_coords: (Seq_len, 2)
        # predicted_tours.unsqueeze(-1) expands (Seq_len,) to (Seq_len, 1)
        # .expand(-1, -1, 2) expands to (Seq_len, 1, 2) -- this is for gather.
        # However, for single instance, simpler numpy indexing might be more direct.
        # Let's use torch.gather consistent with the batch implementation for clarity
        # but apply it after unsqueezing input_coords for the gather op.
        
        # Original input_coords is (Seq_len, 2)
        # We need to gather (Seq_len, 2) from (Seq_len, 2) using (Seq_len, 1) indices
        # So expand indices to (Seq_len, 2) for gather.
        gather_indices_coords = predicted_tours.unsqueeze(-1).expand(-1, 2) # (Seq_len, 2)
        
        # input_coords needs to be unsqueezed at dim 0 for gather to work as desired here
        # torch.gather(input_coords_batched, 1, predicted_tours_batched.unsqueeze(-1).expand(-1, -1, 2))
        # No, input_coords in this function is (seq_len, 2) not (B, S, 2)
        # Let's adjust gathering to work on the single instance
        
        # Simplest way: use numpy indexing after converting to numpy
        inferred_tour_indices = predicted_tours.cpu().numpy().tolist() # Convert to list
        
        # Get coordinates in the inferred order
        original_coords_np = input_coords.cpu().numpy() # (Seq_len, 2)
        inferred_tour_coords_np = original_coords_np[inferred_tour_indices] # (Seq_len, 2)
        
        # Calculate reward (tour length)
        # Complete the cycle by adding the first point again
        path_coords = np.vstack([inferred_tour_coords_np, inferred_tour_coords_np[0]])
        
        # Calculate Euclidean distances between consecutive points
        distances = np.linalg.norm(path_coords[1:] - path_coords[:-1], axis=1)
        tour_length = distances.sum()
        # tour_reward = -tour_length # Reward is negative tour length

    return inferred_tour_indices, inferred_tour_coords_np, tour_length

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

def plot_tour(original_coords, inferred_tour_coords, tour_length):
    """
    Plots the inferred TSP tour, marking the visiting order, start, and end points.

    Args:
        original_coords (torch.Tensor or np.ndarray): The original coordinates of all cities. (Seq_len, 2)
        inferred_tour_coords (np.ndarray): The coordinates of the cities in the inferred tour order. (Seq_len, 2)
        tour_length (float): The  tour length.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot all original cities as faint points
    ax.scatter(original_coords[:, 0], original_coords[:, 1], color='lightgray', marker='o', s=50, zorder=1, label='All Taksk Points')

    # Plot the inferred tour path
    # Add the starting point at the end to close the cycle for plotting
    plot_path_coords = np.vstack([inferred_tour_coords, inferred_tour_coords[0]])
    ax.plot(plot_path_coords[:, 0], plot_path_coords[:, 1], 'b-o', linewidth=1.5, markersize=8, label='Inferred Tour Path', zorder=2)

    # Mark the visiting order and points
    for i, (x, y) in enumerate(inferred_tour_coords):
        ax.text(x + 0.01, y + 0.01, str(i + 1), fontsize=9, ha='center', va='center', color='black', weight='bold', zorder=3)
        ax.scatter(x, y, color='blue', s=100, zorder=3) # Highlight visited cities

    # Mark start point (first city in the tour)
    start_x, start_y = inferred_tour_coords[0]
    ax.plot(start_x, start_y, 'go', markersize=12, label='Start Point', zorder=4)
    ax.text(start_x + 0.02, start_y + 0.02, 'S', fontsize=12, color='darkgreen', weight='bold', ha='center', va='center', zorder=5)

    # Mark end point (which is the same as start point for a closed tour)
    # The last city before returning to start point
    end_x, end_y = inferred_tour_coords[-1]
    ax.plot(end_x, end_y, 'ro', markersize=12, label='End Point', zorder=4)
    ax.text(end_x - 0.02, end_y - 0.02, 'E', fontsize=12, color='darkred', weight='bold', ha='center', va='center', zorder=5)


    ax.set_title(f'Inferred Robot Tour (Length: {tour_length:.4f})')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()
    plt.tight_layout()
    plt.show()

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

    print(f"\nStarting LNS. Initial Tour Cost: {current_cost:.4f}")

    for iteration in tqdm(range(max_iterations), desc="LNS Progress"):
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
        
    print(f"LNS Finished. Best Tour Cost: {best_tour_cost:.4f}")
    return best_tour_indices, best_tour_cost

# ----- Side-by-Side Plotting Function -----
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
    
    ax1.set_title(f'RL Model Output (Length: {rl_tour_reward:.4f})')
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



# --- Main Execution Block ---
if __name__ == '__main__':
    # Hyperparameters (must match training and ONNX export)
    embedding_dim = 256
    hidden_dim = 256
    seq_len = 20 # Number of cities (including robot station at index 0)

    # Path to your trained PyTorch model
    pth_model_path = 'tsp_ac_256_2L.pth' 

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
        print(f"  RL Model Tour Length: {rl_tour_reward:.4f}")
        print(f"  LNS Optimized Tour Length: {-lns_best_tour_reward:.4f}")
        if abs(lns_best_tour_cost) < abs(rl_tour_reward):
             print(f"  LNS improved tour by: {abs(rl_tour_reward) - (-lns_best_tour_reward):.4f}")
        else:
             print("  LNS did not find a better solution (or found one of similar quality).")

    except FileNotFoundError:
        print(f"Error: Model file not found at {pth_model_path}.")
        print("Please ensure you have run the training script and the model file exists.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")