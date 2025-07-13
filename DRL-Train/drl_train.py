import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

# ----- Actor (Unified Class) -----
class Actor(nn.Module):
    def __init__(self, embedding_dim=128, hidden_dim=128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Encoder part
        self.ff_layer = nn.Linear(2, embedding_dim)
        self.gru_encoder = nn.GRU(input_size=embedding_dim, hidden_size=embedding_dim, batch_first=True)

        # Decoder part (which includes Attention)
        self.gru_decoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.attention = self._Attention(hidden_dim) # Using inner Attention class

    # Inner Attention class (private to Actor, or could be a separate helper)
    class _Attention(nn.Module):
        def __init__(self, hidden_dim):
            super().__init__()
            self.W1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.v = nn.Linear(hidden_dim, 1, bias=False)

        def forward(self, decoder_hidden, encoder_outputs, mask):
            # decoder_hidden: (1, B, H)
            # encoder_outputs: (B, S, H)
            
            # (1, B, H) -> (B, 1, H) for broadcasting with encoder_outputs
            decoder_hidden_expanded = decoder_hidden.transpose(0, 1) 
            
            # Calculate score: (B, S, H) + (B, 1, H) -> (B, S, H) -> (B, S, 1)
            score = self.v(torch.tanh(self.W1(encoder_outputs) + self.W2(decoder_hidden_expanded)))
            score = score.squeeze(-1) # (B, S)

            # Mask visited (B, S)
            score[mask == 0] = -1e9 
            
            attn_weights = F.softmax(score, dim=-1) # (B, S)
            return attn_weights

    def forward(self, input_coords, seq_len):
        # input_coords: (B, S, 2)

        # Encoder Pass
        # embedded: (B, S, E)
        # encoder_outputs: (B, S, H)
        # encoder_hidden: (1, B, H)
        embedded = self.ff_layer(input_coords)
        encoder_outputs, encoder_hidden = self.gru_encoder(embedded)

        batch_size = input_coords.size(0)
        
        # Initial decoder input: Embedding of the robot station (which is the first city, index 0)
        # 'embedded' is (B, S, E), so embedded[:, 0, :] is (B, E). Unsqueeze(1) makes it (B, 1, E).
        decoder_input = embedded[:, 0, :].unsqueeze(1) # (B, 1, E)
        
        hidden = encoder_hidden # Initial decoder hidden state from encoder
        
        # Initialize mask: Mark the first city (robot station, index 0) as visited
        mask = torch.ones(batch_size, seq_len, device=input_coords.device)
        mask[:, 0] = 0 # Robot station (index 0) is already visited

        log_probs_batch = [] 
        # The tour starts with the robot station (index 0) for all batches
        tours_batch = [torch.zeros(batch_size, dtype=torch.long, device=input_coords.device)] 

        # The loop runs for seq_len - 1 steps to select the remaining cities
        for _ in range(seq_len - 1): # We need to select seq_len - 1 other cities
            output, hidden = self.gru_decoder(decoder_input, hidden)
            attn_weights = self.attention(hidden, encoder_outputs, mask)
            
            dist = torch.distributions.Categorical(attn_weights)
            selected = dist.sample() # (B,)
            
            log_prob = dist.log_prob(selected) # (B,)
            log_probs_batch.append(log_prob)
            
            tours_batch.append(selected)
            
            # Update mask: Set selected nodes to 0 for each item in the batch
            # selected.unsqueeze(-1) changes (B,) to (B, 1) for scatter_
            mask.scatter_(1, selected.unsqueeze(-1), 0)
            
            # --- CORRECT FIX for torch.gather index shape ---
            # selected: (B,)
            # unsqueeze(1) -> (B, 1)
            # unsqueeze(2) -> (B, 1, 1)
            # expand(-1, 1, self.embedding_dim) -> (B, 1, embedding_dim)
            # This 'gather_indices' tensor now has the same number of dimensions as encoder_outputs
            # and is correctly shaped to select an embedding vector for each batch item.
            gather_indices = selected.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.embedding_dim)
            decoder_input = torch.gather(encoder_outputs, 1, gather_indices).detach()
            # decoder_input will be (B, 1, E) as required for the next GRU step
            # --- END CORRECT FIX ---

        tours_batch_tensor = torch.stack(tours_batch, dim=1) # (B, seq_len)
        log_probs_batch_tensor = torch.stack(log_probs_batch, dim=0) # (seq_len, B)

        return log_probs_batch_tensor, tours_batch_tensor, encoder_outputs 


# ----- Critic -----
class Critic(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, encoder_outputs):
        pooled = encoder_outputs.mean(dim=1) # (B, H)
        return self.fc(pooled).squeeze(-1) # (B,)

# ----- Training -----

def train_drl_ac(embedding_dim, hidden_dim, seq_len, total_instances, batch_size, lr, epochs):
    torch.manual_seed(42)

    actor = Actor(embedding_dim, hidden_dim)
    critic = Critic(hidden_dim)

    actor_optim = optim.Adam(actor.parameters(), lr=lr)
    critic_optim = optim.Adam(critic.parameters(), lr=lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor.to(device)
    critic.to(device)

    # Calculate the number of batches
    num_batches_per_total_instances = (total_instances + batch_size - 1) // batch_size # Ceiling division

    print(f"Total instances: {total_instances}, Batch size: {batch_size}")
    print(f"This implies {num_batches_per_total_instances} batches, each trained for {epochs} epochs.")
    print(f"Starting training...")

    # Outer loop: Iterate through the total number of distinct batches
    # Each iteration generates a NEW batch and trains it for 'epochs' times
    for batch_iter_idx in range(num_batches_per_total_instances):
        # Define the fixed robot station coordinates (e.g., at [0.5, 0.5])
        # Ensure it's on the correct device and expanded to match the batch_size
        robot_station_coords_fixed = torch.tensor([[0.55, 0.85]], dtype=torch.float32, device=device).expand(batch_size, 1, 2)

        # Generate random coordinates for the remaining (seq_len - 1) cities
        random_other_coords = torch.rand(batch_size, seq_len - 1, 2, device=device)

        # Concatenate the robot station as the first city for all instances in the batch
        input_coords = torch.cat([robot_station_coords_fixed, random_other_coords], dim=1)


        # Inner loop: Train this specific batch for 'epochs' times
        for epoch_idx in range(epochs):
            # Actor forward pass generates tours and log probabilities for the current batch
            log_probs_batch, tours_batch, encoder_outputs = actor(input_coords, seq_len)

            # Calculate rewards for the current batch of tours
            # input_coords: (B, S, 2)
            # tours_batch: (B, seq_len) - indices
            
            # gathered_coords needs to gather along dim 1 of input_coords (which has shape 2)
            # tours_batch needs to be expanded to (B, seq_len, 2) to gather both x and y coords
            gather_indices_coords = tours_batch.unsqueeze(-1).expand(-1, -1, 2)
            gathered_coords = torch.gather(input_coords, 1, gather_indices_coords)
            
            # Append starting node to complete the cycle for each tour in the batch
            first_nodes = gathered_coords[:, 0, :].unsqueeze(1) 
            paths = torch.cat([gathered_coords, first_nodes], dim=1)
            
            # Calculate distances for each segment in each tour
            segment_distances = torch.norm(paths[:, 1:, :] - paths[:, :-1, :], dim=2)
            
            # Sum distances for each tour to get total reward (negative of total distance)
            rewards = -segment_distances.sum(dim=1)

            # Critic predicts value for the batch
            value = critic(encoder_outputs)
            
            # Calculate advantage for each item in the batch
            advantage = rewards.detach() - value
            
            # Actor Loss: Sum of log_probs (for each tour) multiplied by advantage (for each tour), then mean over batch
            actor_loss = (-log_probs_batch.sum(dim=0) * advantage.detach()).mean()

            # Critic Loss: MSE between predicted value and actual reward (detached)
            critic_loss = (value - rewards.detach()).pow(2).mean()

            # Optimization steps
            critic_optim.zero_grad()
            critic_loss.backward(retain_graph=True) 
            critic_optim.step()
            
            actor_optim.zero_grad()
            actor_loss.backward()
            actor_optim.step()

        # Logging after each 'num_batches_per_total_instances' iteration (which includes 'epochs' steps)
        # You might want more granular logging, but for now, let's log the last epoch's results for this batch
        if (batch_iter_idx + 1) % 50 == 0 or batch_iter_idx == num_batches_per_total_instances - 1: # Log every 50 batch iterations or at the end
            print(f"Batch Iteration {batch_iter_idx+1}/{num_batches_per_total_instances} (Epochs done for this batch): "
                  f"Actor Loss: {actor_loss.item():.4f}, "
                  f"Critic Loss: {critic_loss.item():.4f}, "
                  f"Mean Reward: {rewards.mean().item():.4f}")


    # Save the trained models
    torch.save({
        'actor_state_dict': actor.state_dict(),
        'critic_state_dict': critic.state_dict(),
        'actor_optimizer_state_dict': actor_optim.state_dict(),
        'critic_optimizer_state_dict': critic_optim.state_dict()
    }, 'tsp_ac_256_2L.pth')


# Example Usage:
if __name__ == '__main__':
    # Hyperparameters
    embedding_dim = 256
    hidden_dim = 256
    seq_len = 50  # Number of cities in TSP
    total_training_instances = 200000 # Your total number of instances
    batch_size = 256 # Size of each 'mini-batch' that gets trained for 'epochs'
    lr = 5e-4
    epochs_per_batch = 20 # Number of times each generated batch is trained

    print(f"Starting training with total_instances={total_training_instances}, "
          f"batch_size={batch_size}, "
          f"epochs_per_batch={epochs_per_batch}.")
    
    train_drl_ac(embedding_dim, hidden_dim, seq_len, 
                 total_training_instances, batch_size, lr, epochs_per_batch)
    
    print("Training complete and model saved.")