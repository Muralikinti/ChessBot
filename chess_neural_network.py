import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List
import random

class ChessNet(nn.Module):
    """
    Neural Network for Chess Position Evaluation and Move Prediction
    
    Architecture:
    - Input: 8x8x12 board representation (6 piece types × 2 colors)
    - Convolutional layers for spatial pattern recognition
    - Residual blocks for deep learning
    - Dual output heads: value (position evaluation) + policy (move probabilities)
    """
    
    def __init__(self, num_filters=256, num_residual_blocks=10):
        super(ChessNet, self).__init__()
        
        # Input: 8x8x12 (board representation)
        self.conv_input = nn.Conv2d(12, num_filters, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_filters)
        
        # Residual blocks for deep learning
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_residual_blocks)
        ])
        
        # Value head (position evaluation)
        self.value_head = nn.Sequential(
            nn.Conv2d(num_filters, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()  # Output between -1 and 1
        )
        
        # Policy head (move probabilities)
        # 64*64 = 4096 possible moves (from any square to any square)
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 32, 4096),
            nn.LogSoftmax(dim=1)
        )
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Tensor of shape (batch_size, 12, 8, 8) - board representations
            
        Returns:
            value: Tensor of shape (batch_size, 1) - position evaluations
            policy: Tensor of shape (batch_size, 4096) - move probabilities
        """
        # Input convolution
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Dual heads
        value = self.value_head(x)
        policy = self.policy_head(x)
        
        return value, policy


class ResidualBlock(nn.Module):
    """Residual block for deep learning with skip connections"""
    
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)
    
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual  # Skip connection
        x = F.relu(x)
        return x


class BoardEncoder:
    """Convert chess board to neural network input format"""
    
    # Piece to index mapping
    PIECE_TO_INDEX = {
        'wP': 0, 'bP': 1,
        'wN': 2, 'bN': 3,
        'wB': 4, 'bB': 5,
        'wR': 6, 'bR': 7,
        'wQ': 8, 'bQ': 9,
        'wK': 10, 'bK': 11
    }
    
    @staticmethod
    def encode_board(game_state) -> np.ndarray:
        """
        Convert board to 8x8x12 numpy array
        
        Args:
            game_state: ChessEngine GameState object
            
        Returns:
            numpy array of shape (12, 8, 8) representing the board
        """
        # Initialize 12 planes (6 piece types × 2 colors)
        encoded = np.zeros((12, 8, 8), dtype=np.float32)
        
        # Fill the planes based on piece positions
        for row in range(8):
            for col in range(8):
                piece = game_state.board[row][col]
                if piece != "--":
                    piece_index = BoardEncoder.PIECE_TO_INDEX[piece]
                    encoded[piece_index, row, col] = 1.0
        
        return encoded
    
    @staticmethod
    def encode_move(from_square: Tuple[int, int], to_square: Tuple[int, int]) -> int:
        """
        Convert a move to an index (0-4095)
        
        Args:
            from_square: (row, col) of source square
            to_square: (row, col) of destination square
            
        Returns:
            move index for neural network policy output
        """
        from_row, from_col = from_square
        to_row, to_col = to_square
        return from_row * 512 + from_col * 64 + to_row * 8 + to_col
    
    @staticmethod
    def decode_move(move_index: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Convert move index back to from/to squares
        
        Args:
            move_index: index from neural network output
            
        Returns:
            ((from_row, from_col), (to_row, to_col))
        """
        from_row = move_index // 512
        remainder = move_index % 512
        from_col = remainder // 64
        remainder = remainder % 64
        to_row = remainder // 8
        to_col = remainder % 8
        return (from_row, from_col), (to_row, to_col)


class ChessNeuralNetworkWrapper:
    """Wrapper class for easy integration with existing chess engine"""
    
    def __init__(self, model_path=None, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ChessNet().to(self.device)
        
        if model_path and torch.cuda.is_available():
            try:
                self.model.load_state_dict(torch.load(model_path))
                print(f"Loaded model from {model_path}")
            except:
                print(f"Could not load model from {model_path}, using random weights")
        
        self.model.eval()
    
    def evaluate_position(self, game_state) -> float:
        """
        Get position evaluation from neural network
        
        Args:
            game_state: ChessEngine GameState object
            
        Returns:
            position evaluation (-1 to 1, where 1 is good for white)
        """
        with torch.no_grad():
            board_tensor = torch.FloatTensor(BoardEncoder.encode_board(game_state)).unsqueeze(0).to(self.device)
            value, _ = self.model(board_tensor)
            return value.item()
    
    def get_move_probabilities(self, game_state) -> List[Tuple[Tuple[int, int], Tuple[int, int], float]]:
        """
        Get move probabilities from neural network
        
        Args:
            game_state: ChessEngine GameState object
            
        Returns:
            List of (from_square, to_square, probability) for valid moves
        """
        with torch.no_grad():
            board_tensor = torch.FloatTensor(BoardEncoder.encode_board(game_state)).unsqueeze(0).to(self.device)
            _, policy = self.model(board_tensor)
            policy_probs = torch.exp(policy).cpu().numpy()[0]
            
            # Get all valid moves
            valid_moves = game_state.get_all_valid_moves()
            move_probs = []
            
            for (from_square, to_square) in valid_moves:
                move_index = BoardEncoder.encode_move(from_square, to_square)
                prob = policy_probs[move_index]
                move_probs.append((from_square, to_square, prob))
            
            # Normalize probabilities for valid moves only
            total_prob = sum(prob for _, _, prob in move_probs)
            if total_prob > 0:
                move_probs = [(fs, ts, prob / total_prob) for fs, ts, prob in move_probs]
            
            return move_probs
    
    def save_model(self, path):
        """Save model weights"""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")


# Test function
def test_neural_network():
    """Test the neural network with a sample board position"""
    print("Testing Chess Neural Network...")
    
    # Import game engine
    from ChessEngine import GameState
    
    # Create test game state
    game = GameState()
    
    # Create neural network
    nn_wrapper = ChessNeuralNetworkWrapper()
    
    # Test position evaluation
    eval_score = nn_wrapper.evaluate_position(game)
    print(f"Position evaluation: {eval_score:.4f}")
    
    # Test move probabilities
    move_probs = nn_wrapper.get_move_probabilities(game)
    print(f"Number of legal moves: {len(move_probs)}")
    
    # Show top 5 moves by probability
    move_probs.sort(key=lambda x: x[2], reverse=True)
    print("\nTop 5 moves by neural network:")
    for i, (from_sq, to_sq, prob) in enumerate(move_probs[:5]):
        print(f"{i+1}. From {from_sq} to {to_sq}: {prob:.4f}")
    
    print("Neural network test completed!")


if __name__ == "__main__":
    test_neural_network() 