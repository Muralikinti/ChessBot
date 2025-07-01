"""
Integration of MCTS + Neural Network with existing Chess Game

This module demonstrates how to integrate the new MCTS + Neural Network AI
with your existing chess game interface.
"""

import pygame
import os
import time
from ChessEngine import GameState
from chess_neural_network import ChessNeuralNetworkWrapper
from mcts import MCTSPlayer
from play_classic import *  # Import all the existing GUI functions

# New height for the advanced UI
ADVANCED_UI_HEIGHT = 100

# Enhanced AI settings
class AISettings:
    """Configuration for different AI modes"""
    
    TRADITIONAL_AI = "traditional"
    MCTS_RANDOM = "mcts_random"
    MCTS_TRAINED = "mcts_trained"
    
    def __init__(self):
        self.ai_mode = self.TRADITIONAL_AI
        self.mcts_simulations = 100
        self.model_path = None
        self.thinking_time = 3.0  # seconds


class EnhancedChessGame:
    """Enhanced chess game with multiple AI options"""
    
    def __init__(self):
        self.ai_settings = AISettings()
        self.mcts_player = None
        self.load_mcts_player()
    
    def load_mcts_player(self):
        """Load MCTS player with neural network"""
        try:
            self.mcts_player = MCTSPlayer(
                neural_network_path=self.ai_settings.model_path,
                simulations=self.ai_settings.mcts_simulations
            )
        except Exception as e:
            self.mcts_player = None
    
    def get_ai_move(self, game_state):
        """Get AI move based on current AI mode"""
        start_time = time.time()
        
        if self.ai_settings.ai_mode == AISettings.TRADITIONAL_AI:
            # Use existing minimax AI
            move = game_state.get_ai_move()
            
        elif self.ai_settings.ai_mode in [AISettings.MCTS_RANDOM, AISettings.MCTS_TRAINED]:
            # Use MCTS AI
            if self.mcts_player:
                move = self.mcts_player.get_move(game_state)
            else:
                move = game_state.get_ai_move()
        else:
            # Default to traditional AI
            move = game_state.get_ai_move()
        
        thinking_time = time.time() - start_time
        
        return move
    
    def get_ai_analysis(self, game_state):
        """Get detailed AI analysis of current position"""
        if self.ai_settings.ai_mode in [AISettings.MCTS_RANDOM, AISettings.MCTS_TRAINED] and self.mcts_player:
            analysis = self.mcts_player.get_analysis(game_state, num_simulations=50)  # Quick analysis
            return analysis
        else:
            # Traditional AI analysis
            eval_score = game_state.evaluate_position()
            return {
                'traditional_eval': eval_score,
                'best_move': game_state.get_ai_move()
            }


def draw_ai_mode_selector(win, mouse_pos, ai_settings):
    """Draw AI mode selection interface"""
    y_start = HEIGHT + BUTTON_HEIGHT + BUTTON_MARGIN + 5
    
    # AI Mode label
    font = pygame.font.SysFont('Arial', 16)
    label = font.render('AI Mode:', True, TEXT_COLOR)
    win.blit(label, (10, y_start))
    
    # Mode buttons
    button_width = 120
    button_height = 25
    button_spacing = 5
    modes = [
        (AISettings.TRADITIONAL_AI, "Traditional"),
        (AISettings.MCTS_RANDOM, "MCTS (Random)"),
        (AISettings.MCTS_TRAINED, "MCTS (Trained)")
    ]
    
    mode_buttons = {}
    x_start = 10
    y_pos = y_start + 20  # Reduced spacing to fit better
    
    for i, (mode_key, mode_name) in enumerate(modes):
        button_rect = pygame.Rect(
            x_start + i * (button_width + button_spacing),
            y_pos,
            button_width,
            button_height
        )
        
        # Color based on selection and hover
        if ai_settings.ai_mode == mode_key:
            color = (0, 150, 0)  # Green for selected
        elif button_rect.collidepoint(mouse_pos):
            color = BUTTON_HOVER
        else:
            color = BUTTON_COLOR
        
        pygame.draw.rect(win, color, button_rect)
        pygame.draw.rect(win, TEXT_COLOR, button_rect, 1)
        
        # Button text
        text = pygame.font.SysFont('Arial', 12).render(mode_name, True, TEXT_COLOR)
        text_rect = text.get_rect(center=button_rect.center)
        win.blit(text, text_rect)
        
        mode_buttons[mode_key] = button_rect
    
    return mode_buttons


def draw_analysis_panel(win, analysis):
    """Draw AI analysis information"""
    panel_x = WIDTH - 200
    panel_y = 10
    panel_width = 190
    panel_height = 150
    
    # Background
    pygame.draw.rect(win, (40, 40, 40), (panel_x, panel_y, panel_width, panel_height))
    pygame.draw.rect(win, (100, 100, 100), (panel_x, panel_y, panel_width, panel_height), 2)
    
    # Title
    font_title = pygame.font.SysFont('Arial', 14, bold=True)
    title = font_title.render('AI Analysis', True, (255, 255, 255))
    win.blit(title, (panel_x + 5, panel_y + 5))
    
    # Analysis content
    font_small = pygame.font.SysFont('Arial', 11)
    y_offset = 25
    
    if 'neural_network_eval' in analysis:
        # MCTS analysis
        eval_text = f"NN Eval: {analysis['neural_network_eval']:.3f}"
        eval_surface = font_small.render(eval_text, True, (255, 255, 255))
        win.blit(eval_surface, (panel_x + 5, panel_y + y_offset))
        y_offset += 15
        
        sim_text = f"Simulations: {analysis.get('simulations', 0)}"
        sim_surface = font_small.render(sim_text, True, (255, 255, 255))
        win.blit(sim_surface, (panel_x + 5, panel_y + y_offset))
        y_offset += 15
        
        moves_text = f"Moves: {analysis.get('total_moves_considered', 0)}"
        moves_surface = font_small.render(moves_text, True, (255, 255, 255))
        win.blit(moves_surface, (panel_x + 5, panel_y + y_offset))
        y_offset += 20
        
        # Top moves
        if 'top_moves' in analysis and analysis['top_moves']:
            top_label = font_small.render('Top Moves:', True, (200, 200, 200))
            win.blit(top_label, (panel_x + 5, panel_y + y_offset))
            y_offset += 15
            
            for i, (move, visits) in enumerate(analysis['top_moves'][:3]):
                if move:
                    from_sq, to_sq = move
                    move_text = f"{i+1}. {from_sq}→{to_sq} ({visits})"
                    move_surface = font_small.render(move_text, True, (180, 180, 180))
                    win.blit(move_surface, (panel_x + 5, panel_y + y_offset))
                    y_offset += 12
    
    elif 'traditional_eval' in analysis:
        # Traditional AI analysis
        eval_text = f"Eval: {analysis['traditional_eval']:.1f}"
        eval_surface = font_small.render(eval_text, True, (255, 255, 255))
        win.blit(eval_surface, (panel_x + 5, panel_y + y_offset))


def enhanced_main():
    """Enhanced main game loop with MCTS integration"""
    global WIN, WINDOW_HEIGHT
    
    # Set the window height for the advanced UI
    WINDOW_HEIGHT = HEIGHT + ADVANCED_UI_HEIGHT
    WIN = pygame.display.set_mode((WIDTH, WINDOW_HEIGHT))

    clock = pygame.time.Clock()
    load_images()
    
    # Create enhanced game instance
    enhanced_game = EnhancedChessGame()
    
    # Game state variables
    in_menu = True
    game_state = None
    selected_square = None
    valid_moves = []
    run = True
    promotion_square = None
    promotion_active = False
    game_mode = PLAYER_VS_AI  # Start with AI mode for MCTS demo
    ai_thinking = False
    show_analysis = False
    current_analysis = {}

    while run:
        clock.tick(60)
        mouse_pos = pygame.mouse.get_pos()

        if in_menu:
            # Draw menu with AI mode selection
            pvp_button, pva_button = draw_menu(WIN, mouse_pos)
            
            # Additional AI info
            info_text = "Try the new MCTS + Neural Network AI!"
            font = pygame.font.SysFont('Arial', 18)
            info_surface = font.render(info_text, True, MENU_TEXT_COLOR)
            info_rect = info_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 200))
            WIN.blit(info_surface, info_rect)
            
            pygame.display.update()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = pygame.mouse.get_pos()
                    if pvp_button.collidepoint(x, y):
                        game_mode = PLAYER_VS_PLAYER
                        game_state = GameState()
                        in_menu = False
                        selected_square = None
                        valid_moves = []
                        promotion_square = None
                        promotion_active = False
                    elif pva_button.collidepoint(x, y):
                        game_mode = PLAYER_VS_AI
                        game_state = GameState()
                        in_menu = False
                        selected_square = None
                        valid_moves = []
                        promotion_square = None
                        promotion_active = False
        else:
            # Game is active
            
            # Adjust window height for controls and AI mode selector
            base_control_height = ADVANCED_UI_HEIGHT
            control_height = base_control_height if not promotion_active else base_control_height + PROMOTION_HEIGHT
            target_window_height = HEIGHT + control_height
            
            if WINDOW_HEIGHT != target_window_height:
                WINDOW_HEIGHT = target_window_height
                WIN = pygame.display.set_mode((WIDTH, WINDOW_HEIGHT))
            
            # Check if it's AI's turn and AI should move
            if (game_mode == PLAYER_VS_AI and not game_state.white_to_move and 
                not game_state.game_over and not game_state.pawn_promotion and not ai_thinking):
                ai_thinking = True
                
                # Get AI move using enhanced system
                ai_move = enhanced_game.get_ai_move(game_state)
                
                if ai_move:
                    start_pos, end_pos = ai_move
                    start_row, start_col = start_pos
                    end_row, end_col = end_pos
                    
                    # Make the move
                    game_state.make_move(start_row, start_col, end_row, end_col)
                    
                    # Update AI move tracking for visual highlighting
                    game_state.ai_move_squares = [(start_row, start_col), (end_row, end_col)]
                
                ai_thinking = False
            
            # Get analysis if requested
            if show_analysis and not ai_thinking:
                current_analysis = enhanced_game.get_ai_analysis(game_state)
            
            # Draw everything
            draw_board(WIN, game_state, selected_square, valid_moves, game_state.ai_move_squares)
            draw_pieces(WIN, game_state)
            
            # Draw UI elements
            back_button = draw_back_button(WIN, mouse_pos)
            restart_button = draw_restart_button(WIN, mouse_pos)
            
            # Draw AI mode selector
            mode_buttons = draw_ai_mode_selector(WIN, mouse_pos, enhanced_game.ai_settings)
            
            # Draw analysis panel
            if show_analysis and current_analysis:
                draw_analysis_panel(WIN, current_analysis)
            
            # Draw promotion menu if needed
            if game_state.pawn_promotion:
                piece_rect = draw_promotion_menu(WIN, 'w' if game_state.white_to_move else 'b')
            
            # Draw AI thinking indicator
            if ai_thinking:
                draw_ai_thinking(WIN)
            
            # Draw game end overlay if game is over
            draw_game_end_overlay(WIN, game_state)
            
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = pygame.mouse.get_pos()
                    
                    # Check UI buttons
                    if back_button.collidepoint(x, y):
                        in_menu = True
                        continue
                    
                    if restart_button.collidepoint(x, y):
                        game_state = GameState()
                        selected_square = None
                        valid_moves = []
                        promotion_square = None
                        promotion_active = False
                        ai_thinking = False
                        current_analysis = {}
                        continue
                    
                    # Check AI mode buttons
                    for mode_key, button_rect in mode_buttons.items():
                        if button_rect.collidepoint(x, y):
                            enhanced_game.ai_settings.ai_mode = mode_key
                            enhanced_game.load_mcts_player()  # Reload MCTS player if needed
                            continue
                    
                    # Handle pawn promotion
                    if game_state.pawn_promotion and y > HEIGHT + 60:
                        piece_size = 60
                        spacing = (WIDTH - (piece_size * 4)) // 5
                        for i, piece in enumerate(['Q', 'R', 'B', 'N']):
                            piece_rect = pygame.Rect(spacing + i * (piece_size + spacing),
                                                  HEIGHT + 60,
                                                  piece_size, piece_size)
                            if piece_rect.collidepoint(x, y):
                                game_state.set_promotion_choice(piece)
                                if promotion_square:
                                    game_state.make_move(promotion_square[0], promotion_square[1],
                                                       promotion_square[2], promotion_square[3])
                                promotion_square = None
                                break
                        continue
                    
                    # Handle board clicks (same as original logic)
                    human_turn = (game_mode == PLAYER_VS_PLAYER or 
                                (game_mode == PLAYER_VS_AI and game_state.white_to_move))
                    
                    if (y < HEIGHT and not game_state.game_over and not ai_thinking and human_turn):
                        col, row = x // SQUARE_SIZE, y // SQUARE_SIZE

                        if selected_square:
                            if game_state.make_move(selected_square[0], selected_square[1], row, col):
                                selected_square = None
                                valid_moves = []
                                game_state.ai_move_squares = []
                                
                                # Immediately update display to show human move
                                draw_board(WIN, game_state, selected_square, valid_moves, game_state.ai_move_squares)
                                draw_pieces(WIN, game_state)
                                back_button = draw_back_button(WIN, mouse_pos)
                                restart_button = draw_restart_button(WIN, mouse_pos)
                                mode_buttons = draw_ai_mode_selector(WIN, mouse_pos, enhanced_game.ai_settings)
                                if show_analysis and current_analysis:
                                    draw_analysis_panel(WIN, current_analysis)
                                if game_state.pawn_promotion:
                                    piece_rect = draw_promotion_menu(WIN, 'w' if game_state.white_to_move else 'b')
                                draw_game_end_overlay(WIN, game_state)
                                pygame.display.update()
                                
                                # Update analysis if showing
                                if show_analysis:
                                    current_analysis = enhanced_game.get_ai_analysis(game_state)
                            else:
                                piece = game_state.board[row][col]
                                if piece != "--" and ((game_state.white_to_move and piece[0] == 'w') or 
                                                   (not game_state.white_to_move and piece[0] == 'b')):
                                    selected_square = (row, col)
                                    valid_moves = game_state.get_valid_moves(row, col)
                                else:
                                    selected_square = None
                                    valid_moves = []
                        else:
                            piece = game_state.board[row][col]
                            if piece != "--" and ((game_state.white_to_move and piece[0] == 'w') or 
                                               (not game_state.white_to_move and piece[0] == 'b')):
                                selected_square = (row, col)
                                valid_moves = game_state.get_valid_moves(row, col)

    pygame.quit()


if __name__ == "__main__":
    enhanced_main() 