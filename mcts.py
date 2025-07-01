import math
import random
import copy
from typing import List, Dict, Optional, Tuple
import numpy as np
from chess_neural_network import ChessNeuralNetworkWrapper


class MCTSNode:
    """Node in the Monte Carlo Tree Search tree"""
    
    def __init__(self, game_state, parent=None, move=None, prior_prob=0.0):
        self.game_state = game_state
        self.parent = parent
        self.move = move  # Move that led to this position
        self.children = {}  # Dict[move: MCTSNode]
        
        # MCTS statistics
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior_prob = prior_prob  # Neural network prior probability
        
        # Game termination
        self.is_terminal = game_state.game_over
        self.terminal_value = None
        
        if self.is_terminal:
            if game_state.checkmate:
                # If it's white's turn and checkmate, black won (-1 for white)
                # If it's black's turn and checkmate, white won (+1 for white)
                self.terminal_value = -1.0 if game_state.white_to_move else 1.0
            else:  # Stalemate
                self.terminal_value = 0.0
    
    def is_expanded(self) -> bool:
        """Check if this node has been expanded (has children)"""
        return len(self.children) > 0
    
    def get_value(self) -> float:
        """Get average value of this node"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def get_ucb_score(self, c_puct: float = 1.0) -> float:
        """
        Calculate UCB (Upper Confidence Bound) score for node selection
        
        Args:
            c_puct: Exploration parameter
            
        Returns:
            UCB score for this node
        """
        if self.visit_count == 0:
            return float('inf')
        
        # UCB1 formula with neural network prior
        exploitation = self.get_value()
        exploration = c_puct * self.prior_prob * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        
        return exploitation + exploration
    
    def select_child(self, c_puct: float = 1.0) -> 'MCTSNode':
        """Select child with highest UCB score"""
        return max(self.children.values(), key=lambda child: child.get_ucb_score(c_puct))
    
    def expand(self, neural_network: ChessNeuralNetworkWrapper):
        """
        Expand this node by adding children for all legal moves
        
        Args:
            neural_network: Neural network for move probability evaluation
        """
        if self.is_terminal:
            return
        
        # Get move probabilities from neural network
        move_probs = neural_network.get_move_probabilities(self.game_state)
        
        for from_square, to_square, prob in move_probs:
            # Create new game state for this move
            new_game_state = copy.deepcopy(self.game_state)
            move_made = new_game_state.make_move(from_square[0], from_square[1], to_square[0], to_square[1])
            
            if move_made:
                move = (from_square, to_square)
                child_node = MCTSNode(new_game_state, parent=self, move=move, prior_prob=prob)
                self.children[move] = child_node
    
    def backup(self, value: float):
        """
        Backup value through the tree
        
        Args:
            value: Value to backup (from perspective of player to move at root)
        """
        self.visit_count += 1
        self.value_sum += value
        
        if self.parent:
            # Flip value for opponent
            self.parent.backup(-value)


class MonteCarloTreeSearch:
    """Monte Carlo Tree Search with Neural Network guidance"""
    
    def __init__(self, neural_network: ChessNeuralNetworkWrapper, c_puct: float = 1.0):
        self.neural_network = neural_network
        self.c_puct = c_puct
    
    def search(self, game_state, num_simulations: int = 800) -> Tuple[Optional[Tuple], List[Tuple]]:
        """
        Perform MCTS search to find the best move
        
        Args:
            game_state: Current game state
            num_simulations: Number of MCTS simulations to perform
            
        Returns:
            best_move: Best move found (from_square, to_square)
            visit_counts: List of (move, visit_count) for analysis
        """
        # Create root node
        root = MCTSNode(game_state)
        
        # Perform simulations
        for _ in range(num_simulations):
            self._simulate(root)
        
        # Select best move based on visit counts
        if not root.children:
            return None, []
        
        # Get visit counts for all moves
        visit_counts = [(move, child.visit_count) for move, child in root.children.items()]
        visit_counts.sort(key=lambda x: x[1], reverse=True)
        
        # Select move with most visits
        best_move = max(root.children.items(), key=lambda item: item[1].visit_count)[0]
        
        return best_move, visit_counts
    
    def _simulate(self, root: MCTSNode):
        """
        Perform one MCTS simulation
        
        Args:
            root: Root node to start simulation from
        """
        node = root
        
        # Selection: traverse down the tree
        while node.is_expanded() and not node.is_terminal:
            node = node.select_child(self.c_puct)
        
        # Expansion: add children if not terminal
        if not node.is_terminal and node.visit_count > 0:
            node.expand(self.neural_network)
            if node.children:
                # Select a child for evaluation
                node = random.choice(list(node.children.values()))
        
        # Evaluation: get value from neural network or terminal state
        if node.is_terminal:
            value = node.terminal_value
        else:
            # Use neural network to evaluate position
            nn_value = self.neural_network.evaluate_position(node.game_state)
            # Convert to perspective of player to move at root
            value = nn_value if root.game_state.white_to_move == node.game_state.white_to_move else -nn_value
        
        # Backup: propagate value up the tree
        node.backup(value)
    
    def get_move_probabilities(self, game_state, num_simulations: int = 800, temperature: float = 1.0) -> List[Tuple]:
        """
        Get move probabilities based on MCTS visit counts
        
        Args:
            game_state: Current game state
            num_simulations: Number of MCTS simulations
            temperature: Temperature for probability distribution (0 = deterministic, higher = more random)
            
        Returns:
            List of (move, probability) tuples
        """
        _, visit_counts = self.search(game_state, num_simulations)
        
        if not visit_counts:
            return []
        
        if temperature == 0:
            # Deterministic: select move with highest visit count
            best_move = max(visit_counts, key=lambda x: x[1])[0]
            return [(best_move, 1.0)]
        
        # Apply temperature to visit counts
        visits = np.array([count for _, count in visit_counts])
        
        if temperature != 1.0:
            visits = visits ** (1.0 / temperature)
        
        # Convert to probabilities
        total_visits = np.sum(visits)
        probabilities = visits / total_visits
        
        return [(move, prob) for (move, _), prob in zip(visit_counts, probabilities)]


class MCTSPlayer:
    """Chess player using MCTS + Neural Network"""
    
    def __init__(self, neural_network_path: str = None, simulations: int = 800, c_puct: float = 1.0):
        self.neural_network = ChessNeuralNetworkWrapper(neural_network_path)
        self.mcts = MonteCarloTreeSearch(self.neural_network, c_puct)
        self.simulations = simulations
    
    def get_move(self, game_state) -> Optional[Tuple]:
        """
        Get best move using MCTS
        
        Args:
            game_state: Current game state
            
        Returns:
            Best move as (from_square, to_square) or None if no moves available
        """
        best_move, visit_counts = self.mcts.search(game_state, self.simulations)
        
        if best_move and visit_counts:
            print(f"MCTS evaluated {len(visit_counts)} moves, best move visits: {visit_counts[0][1]}")
        
        return best_move
    
    def get_analysis(self, game_state, num_simulations: int = None) -> Dict:
        """
        Get detailed analysis of position
        
        Args:
            game_state: Current game state
            num_simulations: Number of simulations (uses default if None)
            
        Returns:
            Dictionary with analysis information
        """
        sims = num_simulations or self.simulations
        best_move, visit_counts = self.mcts.search(game_state, sims)
        
        # Neural network evaluation
        nn_eval = self.neural_network.evaluate_position(game_state)
        
        analysis = {
            'best_move': best_move,
            'neural_network_eval': nn_eval,
            'simulations': sims,
            'top_moves': visit_counts[:5],  # Top 5 moves by visit count
            'total_moves_considered': len(visit_counts)
        }
        
        return analysis


# Test function
def test_mcts():
    """Test MCTS implementation"""
    print("Testing MCTS + Neural Network...")
    
    from ChessEngine import GameState
    
    # Create test game
    game = GameState()
    
    # Create MCTS player
    mcts_player = MCTSPlayer(simulations=100)  # Reduced for quick testing
    
    print("Getting MCTS analysis...")
    analysis = mcts_player.get_analysis(game)
    
    print(f"Best move: {analysis['best_move']}")
    print(f"Neural network evaluation: {analysis['neural_network_eval']:.4f}")
    print(f"Simulations: {analysis['simulations']}")
    print(f"Total moves considered: {analysis['total_moves_considered']}")
    
    print("\nTop moves by visit count:")
    for i, (move, visits) in enumerate(analysis['top_moves']):
        from_sq, to_sq = move
        print(f"{i+1}. From {from_sq} to {to_sq}: {visits} visits")
    
    print("MCTS test completed!")


if __name__ == "__main__":
    test_mcts() 