import pygame
import os
from ChessEngine import GameState

# Initialize pygame
pygame.init()

# Window settings
WIDTH, HEIGHT = 640, 640
ROWS, COLS = 8, 8
SQUARE_SIZE = WIDTH // COLS
BUTTON_HEIGHT = 40
BUTTON_WIDTH = 120
BUTTON_MARGIN = 10

# Height for promotion menu
PROMOTION_HEIGHT = 160  

# Initial window height
WINDOW_HEIGHT = HEIGHT + BUTTON_HEIGHT + BUTTON_MARGIN
WIN = pygame.display.set_mode((WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Chess Game")

# Colors for board
WHITE = (240, 217, 181)
BROWN = (181, 136, 99)
HIGHLIGHT = (124, 252, 0, 128)  
CHECK = (255, 0, 0, 128)  
BUTTON_COLOR = (100, 100, 100)
BUTTON_HOVER = (150, 150, 150)
TEXT_COLOR = (255, 255, 255)
PROMOTION_BG = (50, 50, 50)
GAME_END_BG = (0, 0, 0, 180)  
GAME_END_TEXT = (255, 255, 255)
MENU_BG = (40, 40, 40)
MENU_BUTTON_COLOR = (70, 70, 70)
MENU_BUTTON_HOVER = (100, 100, 100)
MENU_TEXT_COLOR = (255, 255, 255)
AI_MOVE_HIGHLIGHT = (255, 255, 0, 100)  

# Store loaded piece images
PIECES = {}

# Load PNG images from assets folder
def load_images():
    pieces = ['wP', 'bP', 'wR', 'bR', 'wN', 'bN', 'wB', 'bB', 'wQ', 'bQ', 'wK', 'bK']
    for piece in pieces:
        path = os.path.join("assets", f"{piece}.png")
        PIECES[piece] = pygame.transform.scale(pygame.image.load(path), (SQUARE_SIZE, SQUARE_SIZE))

# Draw board squares
def draw_board(win, game_state, selected_square=None, valid_moves=None, ai_move_squares=None):
    for row in range(ROWS):
        for col in range(COLS):
            color = WHITE if (row + col) % 2 == 0 else BROWN
            pygame.draw.rect(win, color, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
            
            # Highlight AI's last move squares
            if ai_move_squares and (row, col) in ai_move_squares:
                s = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                pygame.draw.rect(s, AI_MOVE_HIGHLIGHT, (0, 0, SQUARE_SIZE, SQUARE_SIZE))
                win.blit(s, (col * SQUARE_SIZE, row * SQUARE_SIZE))
            
            # Highlight selected square
            if selected_square and (row, col) == selected_square:
                s = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                pygame.draw.rect(s, HIGHLIGHT, (0, 0, SQUARE_SIZE, SQUARE_SIZE))
                win.blit(s, (col * SQUARE_SIZE, row * SQUARE_SIZE))
            
            # Highlight valid moves
            if valid_moves and (row, col) in valid_moves:
                s = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                pygame.draw.rect(s, HIGHLIGHT, (0, 0, SQUARE_SIZE, SQUARE_SIZE))
                win.blit(s, (col * SQUARE_SIZE, row * SQUARE_SIZE))

    # Highlight king in check
    if game_state.in_check:
        king_location = game_state.white_king_location if game_state.white_to_move else game_state.black_king_location
        s = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(s, CHECK, (0, 0, SQUARE_SIZE, SQUARE_SIZE))
        win.blit(s, (king_location[1] * SQUARE_SIZE, king_location[0] * SQUARE_SIZE))

# Draw all pieces from the board array
def draw_pieces(win, game_state):
    for row in range(ROWS):
        for col in range(COLS):
            piece = game_state.board[row][col]
            if piece != "--":
                win.blit(PIECES[piece], (col * SQUARE_SIZE, row * SQUARE_SIZE))

# Draw promotion menu
def draw_promotion_menu(win, color):
    # Draw background
    pygame.draw.rect(win, PROMOTION_BG, (0, HEIGHT + BUTTON_HEIGHT + BUTTON_MARGIN, WIDTH, PROMOTION_HEIGHT))
    
    # Draw title
    font = pygame.font.SysFont('Arial', 24)
    text = font.render('Choose promotion piece:', True, TEXT_COLOR)
    text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT + BUTTON_HEIGHT + BUTTON_MARGIN + 30))
    win.blit(text, text_rect)
    
    # Draw piece options
    pieces = ['Q', 'R', 'B', 'N']
    piece_size = 60
    spacing = (WIDTH - (piece_size * 4)) // 5
    
    for i, piece in enumerate(pieces):
        piece_code = color + piece
        piece_rect = pygame.Rect(spacing + i * (piece_size + spacing), 
                               HEIGHT + BUTTON_HEIGHT + BUTTON_MARGIN + 60,
                               piece_size, piece_size)
        pygame.draw.rect(win, WHITE, piece_rect)
        piece_img = pygame.transform.scale(PIECES[piece_code], (piece_size, piece_size))
        win.blit(piece_img, piece_rect)
        return piece_rect

# Draw game end overlay (checkmate/stalemate)
def draw_game_end_overlay(win, game_state):
    if game_state.game_over:
        # Create semi-transparent overlay
        overlay = pygame.Surface((WIDTH, HEIGHT))
        overlay.set_alpha(180)
        overlay.fill((0, 0, 0))
        win.blit(overlay, (0, 0))
        
        # Determine message
        if game_state.checkmate:
            message = f"Checkmate! {game_state.winner.title()} Wins!"
        elif game_state.stalemate:
            message = "Stalemate! It's a Draw!"
        else:
            message = "Game Over!"
        
        # Draw message
        font = pygame.font.SysFont('Arial', 48, bold=True)
        text = font.render(message, True, GAME_END_TEXT)
        text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 30))
        win.blit(text, text_rect)
        
        # Draw instruction
        instruction = "Click 'Restart Game' to play again"
        font_small = pygame.font.SysFont('Arial', 24)
        instruction_text = font_small.render(instruction, True, GAME_END_TEXT)
        instruction_rect = instruction_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 30))
        win.blit(instruction_text, instruction_rect)

# Game modes
PLAYER_VS_PLAYER = 0
PLAYER_VS_AI = 1

# Draw menu screen
def draw_menu(win, mouse_pos):
    win.fill(MENU_BG)
    
    # Title
    title_font = pygame.font.SysFont('Arial', 48, bold=True)
    title_text = title_font.render('Chess Game', True, MENU_TEXT_COLOR)
    title_rect = title_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 100))
    win.blit(title_text, title_rect)
    
    # Player vs Player button
    pvp_button = pygame.Rect(WIDTH // 2 - 100, HEIGHT // 2 - 20, 200, 50)
    pvp_color = MENU_BUTTON_HOVER if pvp_button.collidepoint(mouse_pos) else MENU_BUTTON_COLOR
    pygame.draw.rect(win, pvp_color, pvp_button)
    pygame.draw.rect(win, MENU_TEXT_COLOR, pvp_button, 2)
    
    pvp_text = pygame.font.SysFont('Arial', 24).render('Player vs Player', True, MENU_TEXT_COLOR)
    pvp_text_rect = pvp_text.get_rect(center=pvp_button.center)
    win.blit(pvp_text, pvp_text_rect)
    
    # Player vs AI button
    pva_button = pygame.Rect(WIDTH // 2 - 100, HEIGHT // 2 + 50, 200, 50)
    pva_color = MENU_BUTTON_HOVER if pva_button.collidepoint(mouse_pos) else MENU_BUTTON_COLOR
    pygame.draw.rect(win, pva_color, pva_button)
    pygame.draw.rect(win, MENU_TEXT_COLOR, pva_button, 2)
    
    pva_text = pygame.font.SysFont('Arial', 24).render('Player vs AI', True, MENU_TEXT_COLOR)
    pva_text_rect = pva_text.get_rect(center=pva_button.center)
    win.blit(pva_text, pva_text_rect)
    
    # Instructions
    instructions = [
        "Player vs Player: Two human players take turns",
        "Player vs AI: You play as White against the computer"
    ]
    
    for i, instruction in enumerate(instructions):
        inst_text = pygame.font.SysFont('Arial', 16).render(instruction, True, MENU_TEXT_COLOR)
        inst_rect = inst_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 150 + i * 25))
        win.blit(inst_text, inst_rect)
    
    return pvp_button, pva_button

# Draw current game mode indicator
def draw_game_mode_indicator(win, game_mode):
    mode_text = "Player vs Player" if game_mode == PLAYER_VS_PLAYER else "Player vs AI"
    font = pygame.font.SysFont('Arial', 16)
    text = font.render(mode_text, True, TEXT_COLOR)
    text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT + 15))
    win.blit(text, text_rect)

# Draw back to menu button
def draw_back_button(win, mouse_pos):
    button_rect = pygame.Rect(10, HEIGHT + BUTTON_MARGIN, 100, BUTTON_HEIGHT)
    color = BUTTON_HOVER if button_rect.collidepoint(mouse_pos) else BUTTON_COLOR
    pygame.draw.rect(win, color, button_rect)
    
    # Draw button text
    font = pygame.font.SysFont('Arial', 16)
    text = font.render('Back to Menu', True, TEXT_COLOR)
    text_rect = text.get_rect(center=button_rect.center)
    win.blit(text, text_rect)
    
    return button_rect

# Draw restart button
def draw_restart_button(win, mouse_pos):
    button_rect = pygame.Rect(WIDTH - BUTTON_WIDTH - 10, HEIGHT + BUTTON_MARGIN, BUTTON_WIDTH, BUTTON_HEIGHT)
    color = BUTTON_HOVER if button_rect.collidepoint(mouse_pos) else BUTTON_COLOR
    pygame.draw.rect(win, color, button_rect)
    
    # Draw button text
    font = pygame.font.SysFont('Arial', 20)
    text = font.render('Restart Game', True, TEXT_COLOR)
    text_rect = text.get_rect(center=button_rect.center)
    win.blit(text, text_rect)
    
    return button_rect

# Draw AI thinking indicator
def draw_ai_thinking(win):
    font = pygame.font.SysFont('Arial', 24)
    text = font.render('AI is thinking...', True, (255, 255, 0))
    text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    
    # Draw semi-transparent background
    overlay = pygame.Surface((text_rect.width + 40, text_rect.height + 20))
    overlay.set_alpha(200)
    overlay.fill((0, 0, 0))
    win.blit(overlay, (text_rect.x - 20, text_rect.y - 10))
    
    win.blit(text, text_rect)



# Main game loop
def main():
    global WIN, WINDOW_HEIGHT
    clock = pygame.time.Clock()
    load_images()
    
    # Game state variables
    in_menu = True
    game_state = None
    selected_square = None
    valid_moves = []
    run = True
    promotion_square = None
    promotion_active = False
    game_mode = PLAYER_VS_PLAYER
    ai_thinking = False

    while run:
        clock.tick(60)
        mouse_pos = pygame.mouse.get_pos()

        if in_menu:
            # Draw menu
            pvp_button, pva_button = draw_menu(WIN, mouse_pos)
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
            
            # Dynamically resize window for promotion menu
            if game_state.pawn_promotion and not promotion_active:
                WINDOW_HEIGHT = HEIGHT + BUTTON_HEIGHT + BUTTON_MARGIN + PROMOTION_HEIGHT
                WIN = pygame.display.set_mode((WIDTH, WINDOW_HEIGHT))
                promotion_active = True
            elif not game_state.pawn_promotion and promotion_active:
                WINDOW_HEIGHT = HEIGHT + BUTTON_HEIGHT + BUTTON_MARGIN
                WIN = pygame.display.set_mode((WIDTH, WINDOW_HEIGHT))
                promotion_active = False
            
            # Check if it's AI's turn and AI should move
            if (game_mode == PLAYER_VS_AI and not game_state.white_to_move and 
                not game_state.game_over and not game_state.pawn_promotion and not ai_thinking):
                ai_thinking = True

                # Update display before AI thinks
                pygame.display.update()  
                
                # Get AI move
                ai_move = game_state.get_ai_move()
                if ai_move:
                    start_pos, end_pos = ai_move
                    start_row, start_col = start_pos
                    end_row, end_col = end_pos
                    
                    # Make the move
                    game_state.make_move(start_row, start_col, end_row, end_col)
                    
                    # Update AI move tracking for visual highlighting
                    game_state.ai_move_squares = [(start_row, start_col), (end_row, end_col)]
                
                ai_thinking = False
            
            # Draw everything
            draw_board(WIN, game_state, selected_square, valid_moves, game_state.ai_move_squares)
            draw_pieces(WIN, game_state)
            
            # Draw UI elements
            back_button = draw_back_button(WIN, mouse_pos)
            restart_button = draw_restart_button(WIN, mouse_pos)
            draw_game_mode_indicator(WIN, game_mode)
            
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
                    
                    # Check if back button was clicked
                    if back_button.collidepoint(x, y):
                        in_menu = True
                        if game_state:
                            game_state.ai_move_squares = []
                        continue
                    
                    # Check if restart button was clicked
                    if restart_button.collidepoint(x, y):
                        game_state = GameState()
                        game_state.clear_transposition_table()  
                        
                        selected_square = None
                        valid_moves = []
                        promotion_square = None
                        promotion_active = False
                        ai_thinking = False
                        continue
                    
                    # Handle pawn promotion
                    if game_state.pawn_promotion and y > HEIGHT + BUTTON_HEIGHT + BUTTON_MARGIN:
                        piece_size = 60
                        spacing = (WIDTH - (piece_size * 4)) // 5
                        for i, piece in enumerate(['Q', 'R', 'B', 'N']):
                            piece_rect = pygame.Rect(spacing + i * (piece_size + spacing),
                                                  HEIGHT + BUTTON_HEIGHT + BUTTON_MARGIN + 60,
                                                  piece_size, piece_size)
                            if piece_rect.collidepoint(x, y):
                                game_state.set_promotion_choice(piece)
                                if promotion_square:
                                    game_state.make_move(promotion_square[0], promotion_square[1],
                                                       promotion_square[2], promotion_square[3])
                                promotion_square = None
                                break
                        continue
                    
                    # Only process board clicks if they're within the board area, game is not over, 
                    # AI is not thinking, and it's human player's turn
                    human_turn = (game_mode == PLAYER_VS_PLAYER or 
                                (game_mode == PLAYER_VS_AI and game_state.white_to_move))
                    
                    if (y < HEIGHT and not game_state.game_over and not ai_thinking and human_turn):
                        col, row = x // SQUARE_SIZE, y // SQUARE_SIZE

                        if selected_square:
                            # Try to make the move
                            if game_state.make_move(selected_square[0], selected_square[1], row, col):
                                selected_square = None
                                valid_moves = []
                                
                                # Clear AI move highlighting when human moves
                                game_state.ai_move_squares = []
                                
                                # Update display immediately to show player's move
                                draw_board(WIN, game_state, selected_square, valid_moves, game_state.ai_move_squares)
                                draw_pieces(WIN, game_state)
                                back_button = draw_back_button(WIN, mouse_pos)
                                restart_button = draw_restart_button(WIN, mouse_pos)
                                draw_game_mode_indicator(WIN, game_mode)
                                if game_state.pawn_promotion:
                                    piece_rect = draw_promotion_menu(WIN, 'w' if game_state.white_to_move else 'b')
                                draw_game_end_overlay(WIN, game_state)
                                pygame.display.update()
                                
                                # Check for pawn promotion
                                piece = game_state.board[row][col]
                                if piece[1] == 'P' and (row == 0 or row == 7):
                                    game_state.pawn_promotion = True
                                    promotion_square = (selected_square[0], selected_square[1], row, col)
                            else:
                                # If move is invalid, select the new square if it's a valid piece
                                piece = game_state.board[row][col]
                                if piece != "--" and ((game_state.white_to_move and piece[0] == 'w') or 
                                                   (not game_state.white_to_move and piece[0] == 'b')):
                                    selected_square = (row, col)
                                    valid_moves = game_state.get_valid_moves(row, col)
                                else:
                                    selected_square = None
                                    valid_moves = []
                        else:
                            # Select a piece if it's the correct color
                            piece = game_state.board[row][col]
                            if piece != "--" and ((game_state.white_to_move and piece[0] == 'w') or 
                                               (not game_state.white_to_move and piece[0] == 'b')):
                                selected_square = (row, col)
                                valid_moves = game_state.get_valid_moves(row, col)

    pygame.quit()

if __name__ == "__main__":
    main()