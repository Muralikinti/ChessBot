import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import copy
import os
import time
from collections import deque
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

from ChessEngine import GameState
from chess_neural_network import ChessNeuralNetworkWrapper, BoardEncoder, ChessNet
from mcts import MCTSPlayer, MonteCarloTreeSearch


class TrainingData:
    """Container for training examples"""
    
    def __init__(self, board_state, move_probs, outcome):
        self.board_state = board_state  # 8x8x12 numpy array
        self.move_probs = move_probs    # Move probabilities from MCTS
        self.outcome = outcome          # Game outcome (-1, 0, 1)


class SelfPlayTrainer:
    """Self-play training system for chess neural network"""
    
    def __init__(self, 
                 model_path: str = None,
                 batch_size: int = 32,
                 learning_rate: float = 0.001,
                 buffer_size: int = 100000):
        
        # Initialize neural network
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.neural_network = ChessNeuralNetworkWrapper(model_path, self.device)
        self.neural_network.model.train()  # Set to training mode
        
        # Training parameters
        self.batch_size = batch_size
        self.optimizer = optim.Adam(self.neural_network.model.parameters(), lr=learning_rate)
        self.criterion_value = nn.MSELoss()
        self.criterion_policy = nn.KLDivLoss(reduction='batchmean')
        
        # Training data buffer
        self.training_buffer = deque(maxlen=buffer_size)
        
        # Training statistics
        self.training_history = {
            'games_played': [],
            'avg_game_length': [],
            'value_loss': [],
            'policy_loss': [],
            'total_loss': []
        }
        
        # Self-play parameters
        self.mcts_simulations = 50  # Reduced for faster training
        self.temperature = 1.0
    
    def generate_self_play_game(self, verbose: bool = False) -> List[TrainingData]:
        """
        Generate one self-play game and return training data
        
        Args:
            verbose: Whether to print game progress
            
        Returns:
            List of training examples from the game
        """
        game = GameState()
        mcts_player = MCTSPlayer(simulations=self.mcts_simulations)
        mcts_player.neural_network = self.neural_network  # Use current network
        
        game_data = []
        move_count = 0
        
        if verbose:
            print("Starting self-play game...")
        
        while not game.game_over and move_count < 200:  # Max 200 moves to prevent infinite games
            # Get current board state
            board_state = BoardEncoder.encode_board(game)
            
            # Get move probabilities from MCTS
            move_probs_list = mcts_player.mcts.get_move_probabilities(
                game, self.mcts_simulations, self.temperature
            )
            
            # Convert to policy vector
            policy_vector = np.zeros(4096, dtype=np.float32)
            for move, prob in move_probs_list:
                from_square, to_square = move
                move_index = BoardEncoder.encode_move(from_square, to_square)
                policy_vector[move_index] = prob
            
            # Store training data (outcome will be filled later)
            game_data.append(TrainingData(board_state, policy_vector, None))
            
            # Select move (with temperature for exploration)
            if move_probs_list:
                if self.temperature == 0:
                    # Greedy selection
                    best_move = max(move_probs_list, key=lambda x: x[1])[0]
                else:
                    # Probabilistic selection
                    moves, probs = zip(*move_probs_list)
                    move_idx = np.random.choice(len(moves), p=probs)
                    best_move = moves[move_idx]
                
                # Make the move
                from_square, to_square = best_move
                game.make_move(from_square[0], from_square[1], to_square[0], to_square[1])
                move_count += 1
                
                if verbose and move_count % 10 == 0:
                    print(f"Move {move_count}: From {from_square} to {to_square}")
            else:
                break
        
        # Determine game outcome
        if game.checkmate:
            # Last player to move won
            outcome = 1.0 if not game.white_to_move else -1.0
        else:
            outcome = 0.0  # Draw/stalemate
        
        # Fill in outcomes for all training data
        for i, data in enumerate(game_data):
            # Alternate perspectives (white = +1, black = -1)
            player_outcome = outcome if i % 2 == 0 else -outcome
            data.outcome = player_outcome
        
        if verbose:
            result = "White wins" if outcome == 1 else "Black wins" if outcome == -1 else "Draw"
            print(f"Game finished: {result} in {move_count} moves")
        
        return game_data
    
    def train_on_batch(self) -> Tuple[float, float, float]:
        """
        Train the neural network on a batch of data
        
        Returns:
            value_loss, policy_loss, total_loss
        """
        if len(self.training_buffer) < self.batch_size:
            return 0.0, 0.0, 0.0
        
        # Sample random batch
        batch = random.sample(self.training_buffer, self.batch_size)
        
        # Prepare batch data
        board_states = torch.FloatTensor([data.board_state for data in batch]).to(self.device)
        target_values = torch.FloatTensor([data.outcome for data in batch]).unsqueeze(1).to(self.device)
        target_policies = torch.FloatTensor([data.move_probs for data in batch]).to(self.device)
        
        # Forward pass
        predicted_values, predicted_policies = self.neural_network.model(board_states)
        
        # Calculate losses
        value_loss = self.criterion_value(predicted_values, target_values)
        policy_loss = self.criterion_policy(predicted_policies, target_policies)
        total_loss = value_loss + policy_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return value_loss.item(), policy_loss.item(), total_loss.item()
    
    def self_play_training_loop(self, 
                               num_games: int = 100,
                               training_interval: int = 10,
                               save_interval: int = 50,
                               save_dir: str = "models"):
        """
        Main self-play training loop
        
        Args:
            num_games: Number of self-play games to generate
            training_interval: Train network every N games
            save_interval: Save model every N games
            save_dir: Directory to save models
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Starting self-play training for {num_games} games...")
        print(f"Training interval: {training_interval}, Save interval: {save_interval}")
        
        start_time = time.time()
        
        for game_num in range(1, num_games + 1):
            print(f"\n--- Game {game_num}/{num_games} ---")
            
            # Generate self-play game
            game_start = time.time()
            game_data = self.generate_self_play_game(verbose=(game_num % 10 == 0))
            game_time = time.time() - game_start
            
            # Add to training buffer
            self.training_buffer.extend(game_data)
            
            # Update statistics
            self.training_history['games_played'].append(game_num)
            self.training_history['avg_game_length'].append(len(game_data))
            
            print(f"Game {game_num} completed in {game_time:.2f}s, {len(game_data)} moves")
            print(f"Training buffer size: {len(self.training_buffer)}")
            
            # Train network
            if game_num % training_interval == 0:
                print("Training neural network...")
                train_start = time.time()
                
                # Multiple training steps per training session
                total_value_loss = 0
                total_policy_loss = 0
                total_loss = 0
                num_batches = min(10, len(self.training_buffer) // self.batch_size)
                
                for _ in range(num_batches):
                    v_loss, p_loss, t_loss = self.train_on_batch()
                    total_value_loss += v_loss
                    total_policy_loss += p_loss
                    total_loss += t_loss
                
                if num_batches > 0:
                    avg_value_loss = total_value_loss / num_batches
                    avg_policy_loss = total_policy_loss / num_batches
                    avg_total_loss = total_loss / num_batches
                    
                    self.training_history['value_loss'].append(avg_value_loss)
                    self.training_history['policy_loss'].append(avg_policy_loss)
                    self.training_history['total_loss'].append(avg_total_loss)
                    
                    train_time = time.time() - train_start
                    print(f"Training completed in {train_time:.2f}s")
                    print(f"Value Loss: {avg_value_loss:.4f}, Policy Loss: {avg_policy_loss:.4f}")
            
            # Save model
            if game_num % save_interval == 0:
                model_path = os.path.join(save_dir, f"chess_model_game_{game_num}.pth")
                self.neural_network.save_model(model_path)
                print(f"Model saved to {model_path}")
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f}s ({total_time/60:.1f} minutes)")
        
        # Save final model
        final_model_path = os.path.join(save_dir, "chess_model_final.pth")
        self.neural_network.save_model(final_model_path)
        
        # Plot training history
        self.plot_training_history()
    
    def plot_training_history(self):
        """Plot training statistics"""
        if not self.training_history['value_loss']:
            print("No training data to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Game length over time
        axes[0, 0].plot(self.training_history['games_played'], self.training_history['avg_game_length'])
        axes[0, 0].set_title('Average Game Length')
        axes[0, 0].set_xlabel('Games Played')
        axes[0, 0].set_ylabel('Moves per Game')
        
        # Training losses
        training_games = [g for g in self.training_history['games_played'] 
                         if g <= len(self.training_history['value_loss']) * 10][:len(self.training_history['value_loss'])]
        
        axes[0, 1].plot(training_games, self.training_history['value_loss'], label='Value Loss')
        axes[0, 1].plot(training_games, self.training_history['policy_loss'], label='Policy Loss')
        axes[0, 1].set_title('Training Losses')
        axes[0, 1].set_xlabel('Games Played')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        
        # Total loss
        axes[1, 0].plot(training_games, self.training_history['total_loss'])
        axes[1, 0].set_title('Total Loss')
        axes[1, 0].set_xlabel('Games Played')
        axes[1, 0].set_ylabel('Loss')
        
        # Buffer size over time
        axes[1, 1].plot(self.training_history['games_played'], 
                       [min(sum(self.training_history['avg_game_length'][:i+1]), len(self.training_buffer)) 
                        for i in range(len(self.training_history['games_played']))])
        axes[1, 1].set_title('Training Buffer Size')
        axes[1, 1].set_xlabel('Games Played')
        axes[1, 1].set_ylabel('Training Examples')
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()
        print("Training history saved to training_history.png")


def quick_training_demo():
    """Quick demo of self-play training"""
    print("=== Self-Play Training Demo ===")
    
    # Create trainer
    trainer = SelfPlayTrainer(
        batch_size=16,
        learning_rate=0.001,
        buffer_size=1000
    )
    
    # Run a few games for demonstration
    trainer.self_play_training_loop(
        num_games=20,
        training_interval=5,
        save_interval=10
    )
    
    print("Demo completed!")


def start_full_training():
    """Start full self-play training"""
    print("=== Starting Full Self-Play Training ===")
    
    trainer = SelfPlayTrainer(
        batch_size=32,
        learning_rate=0.001,
        buffer_size=50000
    )
    
    # Run full training
    trainer.self_play_training_loop(
        num_games=500,
        training_interval=10,
        save_interval=50
    )


if __name__ == "__main__":
    # Run quick demo by default
    quick_training_demo() 