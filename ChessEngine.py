class GameState():
    def __init__(self):
        self.board = [
            ["bR", "bN", "bB", "bQ", "bK", "bB", "bN", "bR"],
            ["bP", "bP", "bP", "bP", "bP", "bP", "bP", "bP"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["wP", "wP", "wP", "wP", "wP", "wP", "wP", "wP"],
            ["wR", "wN", "wB", "wQ", "wK", "wB", "wN", "wR"]
        ]
        self.white_to_move = True
        self.move_log = []
        self.white_king_location = (7, 4)  
        self.black_king_location = (0, 4)  
        self.in_check = False
        self.pins = []  
        self.checks = []  
        
        # Castling rights
        self.white_king_moved = False
        self.black_king_moved = False
        self.white_rook_king_moved = False
        self.white_rook_queen_moved = False
        self.black_rook_king_moved = False
        self.black_rook_queen_moved = False
        
        # En passant
        self.en_passant_possible = None  
        
        # Pawn promotion
        self.pawn_promotion = False
        self.promotion_choice = 'Q'  
        
        # Game state
        self.game_over = False
        self.checkmate = False
        self.stalemate = False
        self.winner = None  
        
        # AI settings
        self.ai_depth = 3  
        self.ai_move_squares = []  
        self.transposition_table = {}  

        # Opening book
        self.opening_book = self.initialize_opening_book()
        
        # Enable/disable opening book
        self.use_opening_book = True  
        self.opening_moves_played = 0  

    # Piece values for evaluation
    PIECE_VALUES = {
        'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 20000
    }
    
    # Position bonus tables (simplified)
    PAWN_TABLE = [
        [0,  0,  0,  0,  0,  0,  0,  0],
        [50, 50, 50, 50, 50, 50, 50, 50],
        [10, 10, 20, 30, 30, 20, 10, 10],
        [5,  5, 10, 25, 25, 10,  5,  5],
        [0,  0,  0, 20, 20,  0,  0,  0],
        [5, -5,-10,  0,  0,-10, -5,  5],
        [5, 10, 10,-20,-20, 10, 10,  5],
        [0,  0,  0,  0,  0,  0,  0,  0]
    ]
    
    KNIGHT_TABLE = [
        [-50,-40,-30,-30,-30,-30,-40,-50],
        [-40,-20,  0,  0,  0,  0,-20,-40],
        [-30,  0, 10, 15, 15, 10,  0,-30],
        [-30,  5, 15, 20, 20, 15,  5,-30],
        [-30,  0, 15, 20, 20, 15,  0,-30],
        [-30,  5, 10, 15, 15, 10,  5,-30],
        [-40,-20,  0,  5,  5,  0,-20,-40],
        [-50,-40,-30,-30,-30,-30,-40,-50]
    ]
    
    BISHOP_TABLE = [
        [-20,-10,-10,-10,-10,-10,-10,-20],
        [-10,  0,  0,  0,  0,  0,  0,-10],
        [-10,  0,  5, 10, 10,  5,  0,-10],
        [-10,  5,  5, 10, 10,  5,  5,-10],
        [-10,  0, 10, 10, 10, 10,  0,-10],
        [-10, 10, 10, 10, 10, 10, 10,-10],
        [-10,  5,  0,  0,  0,  0,  5,-10],
        [-20,-10,-10,-10,-10,-10,-10,-20]
    ]
    
    ROOK_TABLE = [
        [0,  0,  0,  5,  5,  0,  0,  0],
        [-5, 0,  0,  0,  0,  0,  0, -5],
        [-5, 0,  0,  0,  0,  0,  0, -5],
        [-5, 0,  0,  0,  0,  0,  0, -5],
        [-5, 0,  0,  0,  0,  0,  0, -5],
        [-5, 0,  0,  0,  0,  0,  0, -5],
        [5, 10, 10, 10, 10, 10, 10,  5],
        [0,  0,  0,  5,  5,  0,  0,  0]
    ]
    
    QUEEN_TABLE = [
        [-20,-10,-10, -5, -5,-10,-10,-20],
        [-10,  0,  0,  0,  0,  0,  0,-10],
        [-10,  0,  5,  5,  5,  5,  0,-10],
        [ -5,  0,  5,  5,  5,  5,  0, -5],
        [  0,  0,  5,  5,  5,  5,  0, -5],
        [-10,  5,  5,  5,  5,  5,  0,-10],
        [-10,  0,  5,  0,  0,  0,  0,-10],
        [-20,-10,-10, -5, -5,-10,-10,-20]
    ]
    
    KING_MIDDLEGAME_TABLE = [
        [-30,-40,-40,-50,-50,-40,-40,-30],
        [-30,-40,-40,-50,-50,-40,-40,-30],
        [-30,-40,-40,-50,-50,-40,-40,-30],
        [-30,-40,-40,-50,-50,-40,-40,-30],
        [-20,-30,-30,-40,-40,-30,-30,-20],
        [-10,-20,-20,-20,-20,-20,-20,-10],
        [ 20, 20,  0,  0,  0,  0, 20, 20],
        [ 20, 30, 10,  0,  0, 10, 30, 20]
    ]
    
    KING_ENDGAME_TABLE = [
        [-50,-40,-30,-20,-20,-30,-40,-50],
        [-30,-20,-10,  0,  0,-10,-20,-30],
        [-30,-10, 20, 30, 30, 20,-10,-30],
        [-30,-10, 30, 40, 40, 30,-10,-30],
        [-30,-10, 30, 40, 40, 30,-10,-30],
        [-30,-10, 20, 30, 30, 20,-10,-30],
        [-30,-30,  0,  0,  0,  0,-30,-30],
        [-50,-30,-30,-30,-30,-30,-30,-50]
    ]

    def get_valid_moves(self, start_row, start_col):
        piece = self.board[start_row][start_col]
        if piece == "--" or (self.white_to_move and piece[0] != 'w') or (not self.white_to_move and piece[0] != 'b'):
            return []

        # Get all possible moves
        valid_moves = []
        piece_type = piece[1]

        if piece_type == 'P':  
            valid_moves.extend(self.get_pawn_moves(start_row, start_col))
        elif piece_type == 'R':  
            valid_moves.extend(self.get_rook_moves(start_row, start_col))
        elif piece_type == 'N':  
            valid_moves.extend(self.get_knight_moves(start_row, start_col))
        elif piece_type == 'B':  
            valid_moves.extend(self.get_bishop_moves(start_row, start_col))
        elif piece_type == 'Q':  
            valid_moves.extend(self.get_queen_moves(start_row, start_col))
        elif piece_type == 'K':  
            valid_moves.extend(self.get_king_moves(start_row, start_col))
            
            valid_moves.extend(self.get_castle_moves(start_row, start_col))

        # Filter out moves that would put/leave the king in check
        return self.filter_valid_moves(start_row, start_col, valid_moves)

    def filter_valid_moves(self, start_row, start_col, moves):
        valid_moves = []
        for end_row, end_col in moves:
            # Make the move
            piece = self.board[start_row][start_col]
            captured_piece = self.board[end_row][end_col]
            self.board[end_row][end_col] = piece
            self.board[start_row][start_col] = "--"

            # Update king location if king is moved
            if piece == "wK":
                self.white_king_location = (end_row, end_col)
                self.white_king_moved = True
            elif piece == "bK":
                self.black_king_location = (end_row, end_col)
                self.black_king_moved = True

            # Check if the move leaves the king in check
            if not self.square_under_attack(self.white_king_location if self.white_to_move else self.black_king_location):
                valid_moves.append((end_row, end_col))

            # Undo the move
            self.board[start_row][start_col] = piece
            self.board[end_row][end_col] = captured_piece

            # Restore king location if king was moved
            if piece == "wK":
                self.white_king_location = (start_row, start_col)
                self.white_king_moved = False
            elif piece == "bK":
                self.black_king_location = (start_row, start_col)
                self.black_king_moved = False

        return valid_moves

    def square_under_attack(self, square):
        """Check if a square is under attack by any opponent piece"""
        row, col = square
        opponent_color = 'b' if self.white_to_move else 'w'

        # Check all opponent pieces that could attack this square
        for r in range(8):
            for c in range(8):
                piece = self.board[r][c]
                if piece != "--" and piece[0] == opponent_color:
                    # Get all possible moves for this piece
                    if piece[1] == 'P':  
                        moves = self.get_pawn_moves(r, c)
                    elif piece[1] == 'R':  
                        moves = self.get_rook_moves(r, c)
                    elif piece[1] == 'N':  
                        moves = self.get_knight_moves(r, c)
                    elif piece[1] == 'B':  
                        moves = self.get_bishop_moves(r, c)
                    elif piece[1] == 'Q':  
                        moves = self.get_queen_moves(r, c)
                    elif piece[1] == 'K':  
                        moves = self.get_king_moves(r, c)
                    else:
                        moves = []

                    # If the square is in the moves list, it's under attack
                    if (row, col) in moves:
                        return True
        return False

    def is_in_check(self):
        """Check if the current player's king is in check"""
        king_location = self.white_king_location if self.white_to_move else self.black_king_location
        return self.square_under_attack(king_location)

    def make_move(self, start_row, start_col, end_row, end_col):
        if (end_row, end_col) in self.get_valid_moves(start_row, start_col):
            piece = self.board[start_row][start_col]
            captured_piece = self.board[end_row][end_col]
            
            # Handle castling
            if piece[1] == 'K' and abs(end_col - start_col) == 2:
                # King-side castling
                if end_col > start_col:
                    self.board[end_row][end_col - 1] = self.board[end_row][7]
                    self.board[end_row][7] = "--"
                # Queen-side castling
                else:
                    self.board[end_row][end_col + 1] = self.board[end_row][0]
                    self.board[end_row][0] = "--"

            # Handle en passant
            if piece[1] == 'P' and (end_row, end_col) == self.en_passant_possible:
                self.board[start_row][end_col] = "--"  

            # Handle pawn promotion
            if piece[1] == 'P' and (end_row == 0 or end_row == 7):
                self.board[end_row][end_col] = piece[0] + self.promotion_choice
            else:
                self.board[end_row][end_col] = piece
            self.board[start_row][start_col] = "--"

            # Update king location if king is moved
            if piece == "wK":
                self.white_king_location = (end_row, end_col)
                self.white_king_moved = True
            elif piece == "bK":
                self.black_king_location = (end_row, end_col)
                self.black_king_moved = True

            # Update rook moved status
            if piece == "wR" and start_col == 7:
                self.white_rook_king_moved = True
            elif piece == "wR" and start_col == 0:
                self.white_rook_queen_moved = True
            elif piece == "bR" and start_col == 7:
                self.black_rook_king_moved = True
            elif piece == "bR" and start_col == 0:
                self.black_rook_queen_moved = True

            # Update en passant possibility
            if piece[1] == 'P' and abs(end_row - start_row) == 2:
                self.en_passant_possible = ((start_row + end_row) // 2, end_col)
            else:
                self.en_passant_possible = None

            self.move_log.append((start_row, start_col, end_row, end_col))
            self.white_to_move = not self.white_to_move
            self.in_check = self.is_in_check()
            self.check_game_end()  
            return True
        return False

    def get_pawn_moves(self, row, col):
        moves = []
        piece = self.board[row][col]
        direction = -1 if piece[0] == 'w' else 1  

        # Forward move
        if 0 <= row + direction < 8 and self.board[row + direction][col] == "--":
            moves.append((row + direction, col))
            # Double move from starting position
            if (row == 6 and piece[0] == 'w') or (row == 1 and piece[0] == 'b'):
                if self.board[row + 2 * direction][col] == "--":
                    moves.append((row + 2 * direction, col))

        # Captures
        for c in [col - 1, col + 1]:
            if 0 <= c < 8 and 0 <= row + direction < 8:
                target = self.board[row + direction][c]
                if target != "--" and target[0] != piece[0]:
                    moves.append((row + direction, c))

        # En passant
        if self.en_passant_possible:
            if (row + direction, col - 1) == self.en_passant_possible or (row + direction, col + 1) == self.en_passant_possible:
                moves.append(self.en_passant_possible)

        return moves

    def get_rook_moves(self, row, col):
        moves = []
        piece = self.board[row][col]
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  

        for d_row, d_col in directions:
            for i in range(1, 8):
                new_row, new_col = row + d_row * i, col + d_col * i
                if not (0 <= new_row < 8 and 0 <= new_col < 8):
                    break
                target = self.board[new_row][new_col]
                if target == "--":
                    moves.append((new_row, new_col))
                elif target[0] != piece[0]:
                    moves.append((new_row, new_col))
                    break
                else:
                    break
        return moves

    def get_knight_moves(self, row, col):
        moves = []
        piece = self.board[row][col]
        knight_moves = [
            (-2, -1), (-2, 1), (-1, -2), (-1, 2),
            (1, -2), (1, 2), (2, -1), (2, 1)
        ]

        for d_row, d_col in knight_moves:
            new_row, new_col = row + d_row, col + d_col
            if 0 <= new_row < 8 and 0 <= new_col < 8:
                target = self.board[new_row][new_col]
                if target == "--" or target[0] != piece[0]:
                    moves.append((new_row, new_col))
        return moves

    def get_bishop_moves(self, row, col):
        moves = []
        piece = self.board[row][col]
        directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

        for d_row, d_col in directions:
            for i in range(1, 8):
                new_row, new_col = row + d_row * i, col + d_col * i
                if not (0 <= new_row < 8 and 0 <= new_col < 8):
                    break
                target = self.board[new_row][new_col]
                if target == "--":
                    moves.append((new_row, new_col))
                elif target[0] != piece[0]:
                    moves.append((new_row, new_col))
                    break
                else:
                    break
        return moves

    def get_queen_moves(self, row, col):
        # Combination of rook and bishop moves
        return self.get_rook_moves(row, col) + self.get_bishop_moves(row, col)

    def get_king_moves(self, row, col):
        moves = []
        piece = self.board[row][col]
        king_moves = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 1),
            (1, -1), (1, 0), (1, 1)
        ]

        for d_row, d_col in king_moves:
            new_row, new_col = row + d_row, col + d_col
            if 0 <= new_row < 8 and 0 <= new_col < 8:
                target = self.board[new_row][new_col]
                if target == "--" or target[0] != piece[0]:
                    moves.append((new_row, new_col))
        return moves

    def get_castle_moves(self, row, col):
        moves = []
        if self.in_check:
            return moves

        # King-side castling
        if (self.white_to_move and not self.white_king_moved and not self.white_rook_king_moved) or \
           (not self.white_to_move and not self.black_king_moved and not self.black_rook_king_moved):
            # Check bounds before accessing
            if col + 2 < 8:
                if self.board[row][col + 1] == "--" and self.board[row][col + 2] == "--":
                    if not self.square_under_attack((row, col + 1)) and not self.square_under_attack((row, col + 2)):
                        moves.append((row, col + 2))

        # Queen-side castling
        if (self.white_to_move and not self.white_king_moved and not self.white_rook_queen_moved) or \
           (not self.white_to_move and not self.black_king_moved and not self.black_rook_queen_moved):
            # Check bounds before accessing
            if col - 3 >= 0:
                if self.board[row][col - 1] == "--" and self.board[row][col - 2] == "--" and self.board[row][col - 3] == "--":
                    if not self.square_under_attack((row, col - 1)) and not self.square_under_attack((row, col - 2)):
                        moves.append((row, col - 2))

        return moves

    def set_promotion_choice(self, choice):
        """Set the piece to promote to (Q, R, B, N)"""
        if choice in ['Q', 'R', 'B', 'N']:
            self.promotion_choice = choice
            self.pawn_promotion = False

    def get_all_valid_moves(self):
        """Get all valid moves for the current player"""
        all_moves = []
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                # Check if this piece belongs to the current player
                if piece != "--" and ((self.white_to_move and piece[0] == 'w') or 
                                    (not self.white_to_move and piece[0] == 'b')):
                    piece_moves = self.get_valid_moves(row, col)
                    for move in piece_moves:
                        all_moves.append(((row, col), move))
        return all_moves

    def is_checkmate(self):
        """Check if the current player is in checkmate"""
        return self.in_check and len(self.get_all_valid_moves()) == 0

    def is_stalemate(self):
        """Check if the current player is in stalemate"""
        return not self.in_check and len(self.get_all_valid_moves()) == 0

    def check_game_end(self):
        """Check if the game has ended and update game state accordingly"""
        if self.is_checkmate():
            self.game_over = True
            self.checkmate = True
            self.winner = 'black' if self.white_to_move else 'white'
        elif self.is_stalemate():
            self.game_over = True
            self.stalemate = True
            self.winner = None  
        else:
            self.game_over = False
            self.checkmate = False
            self.stalemate = False

    def evaluate_position(self):
        """Evaluate the current board position"""
        if self.checkmate:
            return -20000 if self.white_to_move else 20000
        if self.stalemate:
            return 0
            
        score = 0
        
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if piece == "--":
                    continue
                    
                piece_type = piece[1]
                piece_color = piece[0]
                piece_value = self.PIECE_VALUES[piece_type]
                
                # Add positional bonuses
                if piece_type == 'P':
                    if piece_color == 'w':
                        piece_value += self.PAWN_TABLE[row][col]
                    else:
                        piece_value += self.PAWN_TABLE[7-row][col]
                elif piece_type == 'N':
                    if piece_color == 'w':
                        piece_value += self.KNIGHT_TABLE[row][col]
                    else:
                        piece_value += self.KNIGHT_TABLE[7-row][col]
                elif piece_type == 'B':
                    if piece_color == 'w':
                        piece_value += self.BISHOP_TABLE[row][col]
                    else:
                        piece_value += self.BISHOP_TABLE[7-row][col]
                elif piece_type == 'R':
                    if piece_color == 'w':
                        piece_value += self.ROOK_TABLE[row][col]
                    else:
                        piece_value += self.ROOK_TABLE[7-row][col]
                elif piece_type == 'Q':
                    if piece_color == 'w':
                        piece_value += self.QUEEN_TABLE[row][col]
                    else:
                        piece_value += self.QUEEN_TABLE[7-row][col]
                elif piece_type == 'K':
                    # Use different king tables for middlegame vs endgame
                    piece_count = sum(1 for r in self.board for p in r if p != "--")
                    is_endgame = piece_count <= 12  
                    
                    if is_endgame:
                        if piece_color == 'w':
                            piece_value += self.KING_ENDGAME_TABLE[row][col]
                        else:
                            piece_value += self.KING_ENDGAME_TABLE[7-row][col]
                    else:
                        if piece_color == 'w':
                            piece_value += self.KING_MIDDLEGAME_TABLE[row][col]
                        else:
                            piece_value += self.KING_MIDDLEGAME_TABLE[7-row][col]
                
                # Add to score based on color
                if piece_color == 'w':
                    score += piece_value
                else:
                    score -= piece_value
        
        # Add king safety evaluation
        score += self.evaluate_king_safety('w') - self.evaluate_king_safety('b')
        
        return score

    def evaluate_king_safety(self, color):
        """Evaluate king safety for a given color"""
        king_location = self.white_king_location if color == 'w' else self.black_king_location
        king_row, king_col = king_location
        safety_score = 0
        
        # Penalty for king in center during middlegame (rough piece count check)
        piece_count = sum(1 for row in self.board for piece in row if piece != "--")
        is_middlegame = piece_count > 20  
        
        if is_middlegame:
            # King should be castled (on sides) during middlegame
            if color == 'w':
                # White king safer on g1, h1, c1, b1
                if king_row == 7 and king_col in [1, 2, 6, 7]:
                    safety_score += 30  
                elif king_row == 7 and king_col in [3, 4, 5]:
                    safety_score -= 40  
                else:
                    safety_score -= 60  
            else:
                # Black king safer on g8, h8, c8, b8
                if king_row == 0 and king_col in [1, 2, 6, 7]:
                    safety_score += 30  
                elif king_row == 0 and king_col in [3, 4, 5]:
                    safety_score -= 40  
                else:
                    safety_score -= 60  
        
        # Pawn shield bonus - check for pawns in front of castled king
        if color == 'w' and king_row == 7:
            # Check for pawn shield
            pawn_shield_cols = []
            if king_col >= 6:  # Kingside castle
                pawn_shield_cols = [5, 6, 7]
            elif king_col <= 2:  # Queenside castle
                pawn_shield_cols = [0, 1, 2]
            
            for col in pawn_shield_cols:
                if col < 8 and king_row > 0:
                    if self.board[king_row - 1][col] == 'wP':
                        safety_score += 10
                    elif self.board[king_row - 1][col] == '--':
                        safety_score -= 15  # Hole in pawn shield
                        
        elif color == 'b' and king_row == 0:
            # Check for pawn shield
            pawn_shield_cols = []
            if king_col >= 6:  # Kingside castle
                pawn_shield_cols = [5, 6, 7]
            elif king_col <= 2:  # Queenside castle
                pawn_shield_cols = [0, 1, 2]
            
            for col in pawn_shield_cols:
                if col < 8 and king_row < 7:
                    if self.board[king_row + 1][col] == 'bP':
                        safety_score += 10
                    elif self.board[king_row + 1][col] == '--':
                        safety_score -= 15  # Hole in pawn shield
        
        # Piece shelter - bonus for having pieces around the king
        pieces_around_king = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                new_row, new_col = king_row + dr, king_col + dc
                if 0 <= new_row < 8 and 0 <= new_col < 8:
                    piece = self.board[new_row][new_col]
                    if piece != "--" and piece[0] == color:
                        pieces_around_king += 1
        
        safety_score += pieces_around_king * 5  # Bonus for friendly pieces around king
        
        return safety_score

    def get_position_hash(self):
        """Create a hash of the current position for transposition table"""
        # Simple position hash - convert board to string
        position_str = ""
        for row in self.board:
            for piece in row:
                position_str += piece
        position_str += "w" if self.white_to_move else "b"
        position_str += str(self.white_king_moved) + str(self.black_king_moved)
        position_str += str(self.white_rook_king_moved) + str(self.white_rook_queen_moved)
        position_str += str(self.black_rook_king_moved) + str(self.black_rook_queen_moved)
        if self.en_passant_possible:
            position_str += f"{self.en_passant_possible[0]}{self.en_passant_possible[1]}"
        return hash(position_str)

    def minimax(self, depth, alpha, beta, maximizing_player):
        """Minimax algorithm with alpha-beta pruning and transposition table"""
        # Check transposition table
        position_hash = self.get_position_hash()
        if position_hash in self.transposition_table:
            cached_depth, cached_score, cached_move = self.transposition_table[position_hash]
            
            # Cached result is at least as deep
            if cached_depth >= depth:  
                return cached_score, cached_move
        
        if depth == 0 or self.game_over:
            score = self.evaluate_position()
            # Store in transposition table
            self.transposition_table[position_hash] = (depth, score, None)
            return score, None
            
        best_move = None
        
        # Get and order moves for better alpha-beta pruning
        moves = self.get_all_valid_moves()
        moves = self.order_moves(moves)
        
        if maximizing_player:
            max_eval = float('-inf')
            for move in moves:
                # Make the move
                start_pos, end_pos = move
                start_row, start_col = start_pos
                end_row, end_col = end_pos
                
                # Save current state
                saved_state = self.save_state()
                
                # Make the move
                self.make_move(start_row, start_col, end_row, end_col)
                
                # Recursively evaluate
                eval_score, _ = self.minimax(depth - 1, alpha, beta, False)
                
                # Restore state
                self.restore_state(saved_state)
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                    
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  
            
            # Store in transposition table
            self.transposition_table[position_hash] = (depth, max_eval, best_move)
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for move in moves:
                # Make the move
                start_pos, end_pos = move
                start_row, start_col = start_pos
                end_row, end_col = end_pos
                
                # Save current state
                saved_state = self.save_state()
                
                # Make the move
                self.make_move(start_row, start_col, end_row, end_col)
                
                # Recursively evaluate
                eval_score, _ = self.minimax(depth - 1, alpha, beta, True)
                
                # Restore state
                self.restore_state(saved_state)
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                    
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  
            
            # Store in transposition table
            self.transposition_table[position_hash] = (depth, min_eval, best_move)
            return min_eval, best_move

    def order_moves(self, moves):
        """Order moves to improve alpha-beta pruning efficiency"""
        def move_priority(move):
            start_pos, end_pos = move
            start_row, start_col = start_pos
            end_row, end_col = end_pos
            
            priority = 0
            piece = self.board[start_row][start_col]
            target = self.board[end_row][end_col]
            
            # Prioritize captures (Most Valuable Victim - Least Valuable Attacker)
            if target != "--":
                priority += self.PIECE_VALUES[target[1]] * 10  
                priority -= self.PIECE_VALUES[piece[1]]       
            
            # Prioritize pawn promotions
            if piece[1] == 'P' and (end_row == 0 or end_row == 7):
                priority += 800  
            
            # Prioritize castling (king safety)
            if piece[1] == 'K' and abs(end_col - start_col) == 2:
                priority += 50
            
            # Small bonus for center control
            if end_row in [3, 4] and end_col in [3, 4]:
                priority += 5
                
            return priority
        
        # Sort moves by priority (highest first)
        return sorted(moves, key=move_priority, reverse=True)

    def get_opening_move(self):
        """Get a move from the opening book if available"""
        # Stop using opening book after 15 moves
        if not self.use_opening_book or len(self.move_log) > 15:  
            return None
            
        # Convert move log to opening book format
        move_sequence = self.get_move_sequence()
        
        if move_sequence in self.opening_book:
            import random
            possible_moves = self.opening_book[move_sequence]
            
            # Filter for valid moves only
            valid_opening_moves = []
            for move_data, weight in possible_moves:
                start_pos, end_pos = move_data
                start_row, start_col = start_pos
                end_row, end_col = end_pos
                
                # Check if this move is actually valid in current position
                valid_moves = self.get_valid_moves(start_row, start_col)
                if (end_row, end_col) in valid_moves:
                    valid_opening_moves.append((move_data, weight))
            
            if valid_opening_moves:
                # Weighted random selection
                total_weight = sum(weight for _, weight in valid_opening_moves)
                if total_weight > 0:
                    rand_num = random.uniform(0, total_weight)
                    cumulative_weight = 0
                    
                    for move_data, weight in valid_opening_moves:
                        cumulative_weight += weight
                        if rand_num <= cumulative_weight:
                            return move_data
        
        return None
    
    def get_move_sequence(self):
        """Convert current move log to opening book format"""
        sequence = []
        for i in range(0, len(self.move_log), 2):  
            if i < len(self.move_log):
                # White move
                start_row, start_col, end_row, end_col = self.move_log[i]
                sequence.append(((start_row, start_col), (end_row, end_col)))
            
            if i + 1 < len(self.move_log):
                # Black move
                start_row, start_col, end_row, end_col = self.move_log[i + 1]
                sequence.append(((start_row, start_col), (end_row, end_col)))
        
        return tuple(sequence)

    def get_ai_move(self):
        """Get the best move for the AI using opening book or minimax"""
        # First try to get a move from the opening book
        opening_move = self.get_opening_move()
        if opening_move:
            return opening_move
        
        # Fall back to minimax if no opening move available
        _, best_move = self.minimax(self.ai_depth, float('-inf'), float('inf'), self.white_to_move)
        return best_move

    def clear_transposition_table(self):
        """Clear the transposition table to free memory"""
        self.transposition_table.clear()
    
    def toggle_opening_book(self, enabled=None):
        """Enable or disable the opening book"""
        if enabled is None:
            self.use_opening_book = not self.use_opening_book
        else:
            self.use_opening_book = enabled
        return self.use_opening_book
    
    def get_opening_info(self):
        """Get information about current opening position"""
        if not self.use_opening_book:
            return "Opening book disabled"
        
        move_sequence = self.get_move_sequence()
        if move_sequence in self.opening_book:
            possible_moves = self.opening_book[move_sequence]
            return f"Opening position found with {len(possible_moves)} possible moves"
        else:
            return "Position not in opening book"
    
    def add_opening_line(self, move_sequence, possible_moves):
        """Add a new opening line to the book"""
        self.opening_book[move_sequence] = possible_moves

    def save_state(self):
        """Save the current game state"""
        return {
            'board': [row[:] for row in self.board],
            'white_to_move': self.white_to_move,
            'white_king_location': self.white_king_location,
            'black_king_location': self.black_king_location,
            'in_check': self.in_check,
            'white_king_moved': self.white_king_moved,
            'black_king_moved': self.black_king_moved,
            'white_rook_king_moved': self.white_rook_king_moved,
            'white_rook_queen_moved': self.white_rook_queen_moved,
            'black_rook_king_moved': self.black_rook_king_moved,
            'black_rook_queen_moved': self.black_rook_queen_moved,
            'en_passant_possible': self.en_passant_possible,
            'pawn_promotion': self.pawn_promotion,
            'promotion_choice': self.promotion_choice,
            'game_over': self.game_over,
            'checkmate': self.checkmate,
            'stalemate': self.stalemate,
            'winner': self.winner
        }

    def restore_state(self, state):
        """Restore the game state from a saved state"""
        self.board = [row[:] for row in state['board']]
        self.white_to_move = state['white_to_move']
        self.white_king_location = state['white_king_location']
        self.black_king_location = state['black_king_location']
        self.in_check = state['in_check']
        self.white_king_moved = state['white_king_moved']
        self.black_king_moved = state['black_king_moved']
        self.white_rook_king_moved = state['white_rook_king_moved']
        self.white_rook_queen_moved = state['white_rook_queen_moved']
        self.black_rook_king_moved = state['black_rook_king_moved']
        self.black_rook_queen_moved = state['black_rook_queen_moved']
        self.en_passant_possible = state['en_passant_possible']
        self.pawn_promotion = state['pawn_promotion']
        self.promotion_choice = state['promotion_choice']
        self.game_over = state['game_over']
        self.checkmate = state['checkmate']
        self.stalemate = state['stalemate']
        self.winner = state['winner']

    def initialize_opening_book(self):
        """Initialize opening book with popular chess openings"""
        # Opening book structure: move_sequence -> [(from_square, to_square), weight]
        # Each opening maps move sequences to possible next moves with weights
        openings = {
            # Italian Game
            (): [  # Starting position
                (((6, 4), (4, 4)), 30),  # e4 
                (((6, 3), (4, 3)), 25),  # d4 
                (((7, 6), (5, 5)), 15),  # Nf3
                (((6, 2), (4, 2)), 10),  # c4 
            ],
            # After 1.e4
            (((6, 4), (4, 4)),): [
                (((1, 4), (3, 4)), 35),  # e5
                (((1, 2), (3, 2)), 20),  # c5 
                (((1, 4), (2, 4)), 15),  # e6 
                (((1, 2), (2, 2)), 10),  # c6 
            ],
            # Italian Game: 1.e4 e5 2.Nf3
            (((6, 4), (4, 4)), ((1, 4), (3, 4))): [
                (((7, 6), (5, 5)), 40),  # Nf3
                (((7, 1), (5, 2)), 25),  # Nc3
                (((6, 5), (4, 5)), 15),  # f4 
            ],
            # Italian Game: 1.e4 e5 2.Nf3 Nc6
            (((6, 4), (4, 4)), ((1, 4), (3, 4)), ((7, 6), (5, 5))): [
                (((0, 1), (2, 2)), 35),  # Nc6
                (((1, 5), (2, 5)), 25),  # f5 
                (((0, 6), (2, 5)), 20),  # Nf6
            ],
            # Italian Game: 1.e4 e5 2.Nf3 Nc6 3.Bc4
            (((6, 4), (4, 4)), ((1, 4), (3, 4)), ((7, 6), (5, 5)), ((0, 1), (2, 2))): [
                (((7, 5), (4, 2)), 40),  # Bc4 
                (((7, 3), (3, 7)), 25),  # Qh5 
                (((6, 1), (4, 1)), 15),  # b3
            ],
            # Italian Game: 1.e4 e5 2.Nf3 Nc6 3.Bc4 Bc5
            (((6, 4), (4, 4)), ((1, 4), (3, 4)), ((7, 6), (5, 5)), ((0, 1), (2, 2)), ((7, 5), (4, 2))): [
                (((0, 5), (3, 2)), 35),  # Bc5
                (((0, 6), (2, 5)), 25),  # Nf6
                (((1, 5), (3, 5)), 15),  # f5
            ],
            
            # Queen's Gambit
            (((6, 3), (4, 3)),): [  # After 1.d4
                (((1, 3), (3, 3)), 30),  # d5
                (((0, 6), (2, 5)), 25),  # Nf6
                (((1, 5), (2, 5)), 15),  # f5 
            ],
            # Queen's Gambit: 1.d4 d5
            (((6, 3), (4, 3)), ((1, 3), (3, 3))): [
                (((6, 2), (4, 2)), 40),  # c4 
                (((7, 6), (5, 5)), 25),  # Nf3
                (((7, 1), (5, 2)), 15),  # Nc3
            ],
            # Queen's Gambit Accepted: 1.d4 d5 2.c4 dxc4
            (((6, 3), (4, 3)), ((1, 3), (3, 3)), ((6, 2), (4, 2))): [
                (((3, 3), (4, 2)), 50),  # dxc4 
                (((1, 4), (2, 4)), 30),  # e6 
                (((1, 2), (2, 2)), 20),  # c6 
            ],
            
            # Sicilian Defense responses
            (((6, 4), (4, 4)), ((1, 2), (3, 2))): [  # After 1.e4 c5
                (((7, 6), (5, 5)), 40),  # Nf3
                (((7, 1), (5, 2)), 25),  # Nc3 
                (((6, 5), (4, 5)), 15),  # f4 
            ],
            (((6, 4), (4, 4)), ((1, 2), (3, 2)), ((7, 6), (5, 5))): [  # Sicilian: 1.e4 c5 2.Nf3
                (((1, 3), (3, 3)), 30),  # d6
                (((0, 1), (2, 2)), 25),  # Nc6
                (((1, 6), (2, 6)), 20),  # g6 
            ],
        }
        
        return openings
