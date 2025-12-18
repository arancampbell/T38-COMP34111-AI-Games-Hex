import copy
import random
import math
import time
from src.AgentBase import AgentBase
from src.Colour import Colour
from src.Move import Move


class Node:
    def __init__(self, move, parent=None):
        self.move = move
        self.parent = parent
        self.children = {}

        # MC Stats
        self.wins = 0.0
        self.visits = 0

        # RAVE Stats
        self.amaf_wins = 0.0
        self.amaf_visits = 0

        self.untried_moves = None
        self.player_just_moved = None


class MyAgent(AgentBase):
    def __init__(self, colour: Colour, total_time_limit: int = 175.0):
        super().__init__(colour)
        self.root = None
        self.time_limit = 9.0
        self.rave_k = 1000
        self.size = 11

    def make_move(self, turn, board, opp_move):
        # 1. Tree Reuse
        if self.root is not None and opp_move is not None:
            if not opp_move.is_swap():
                opp_move_key = (opp_move.x, opp_move.y)
                if opp_move_key in self.root.children:
                    self.root = self.root.children[opp_move_key]
                    self.root.parent = None
                else:
                    self.root = None
            else:
                self.root = None

        if self.root is None:
            self.root = Node(None)
            self.root.untried_moves = self.get_legal_moves_as_tuples(board)
            self.root.player_just_moved = Colour.opposite(self.colour)

        # 2. Check for Instant Win (Solver)
        # If there is a direct winning move, take it immediately.
        # This prevents MCTS "noise" from ruining an obvious win.
        sim_board = self.board_to_int_matrix(board)
        legal_moves = self.get_legal_moves_as_tuples(board)

        # Only run this check if the board is reasonably full (e.g. >30 moves played)
        if len(legal_moves) < (self.size * self.size) - 30:
            for move in legal_moves:
                self.apply_move_int(sim_board, move, self.colour)
                if self.check_winner_int(sim_board) == self.colour:
                    print(f"Solver found winning move: {move}")
                    return Move(move[0], move[1])
                # Backtrack
                sim_board[move[0]][move[1]] = 0

        # 3. MCTS Loop
        start_time = time.time()
        iterations = 0

        while (time.time() - start_time) < self.time_limit:
            # Re-create simple board for this iteration
            sim_board = self.board_to_int_matrix(board)
            node = self.root
            state_colour = self.colour

            # --- SELECTION ---
            while node.untried_moves == [] and node.children:
                node = self.best_child_rave(node)
                self.apply_move_int(sim_board, node.move, node.player_just_moved)
                state_colour = Colour.opposite(state_colour)

            # --- EXPANSION ---
            if node.untried_moves:
                move_tuple = random.choice(node.untried_moves)
                self.apply_move_int(sim_board, move_tuple, state_colour)
                child = Node(move_tuple, parent=node)
                child.untried_moves = [m for m in node.untried_moves if m != move_tuple]
                child.player_just_moved = state_colour
                node.untried_moves.remove(move_tuple)
                node.children[move_tuple] = child
                node = child

            # --- SIMULATION ---
            # Run simulation and track moves
            sim_turn_colour = Colour.opposite(node.player_just_moved)
            winner_colour, red_moves, blue_moves = self.run_simulation_rave(sim_board, sim_turn_colour)

            # --- BACKPROPAGATION (CORRECTED) ---
            # Identify which moves belonged to the winner
            winning_moves_set = red_moves if winner_colour == Colour.RED else blue_moves

            while node is not None:
                node.visits += 1
                if node.player_just_moved == winner_colour:
                    node.wins += 1

                # CRITICAL FIX: Only update RAVE if the move in the tree
                # matches the colour of the winner in the simulation.
                # If Red won the sim, we only boost Red nodes that played moves in the winning set.
                if node.player_just_moved == winner_colour:
                    if node.move in winning_moves_set:
                        node.amaf_visits += 1
                        node.amaf_wins += 1
                else:
                    # If I am Blue, and Red won, I check if my move was in the LOSING set (Blue moves)
                    # If so, I increment visits but NOT wins (this discourages the move)
                    losing_moves_set = blue_moves if winner_colour == Colour.RED else red_moves
                    if node.move in losing_moves_set:
                        node.amaf_visits += 1
                        # No win increment

                node = node.parent

            iterations += 1

        # 4. Final Selection
        if not self.root.children:
            return self.safe_fallback_move(board)

        best_move_tuple = max(self.root.children, key=lambda k: self.root.children[k].visits)
        best_node = self.root.children[best_move_tuple]

        self.root = best_node
        self.root.parent = None

        print(f"MCTS-RAVE Iterations: {iterations}, Conf: {best_node.wins / best_node.visits:.2f}")
        return Move(best_move_tuple[0], best_move_tuple[1])

    # Keep all Helper Methods (best_child_rave, run_simulation_rave, etc.) exactly as before.
    # ... [Insert helper methods here] ...

    # --------------------------------------------------------
    # COPY PASTE THE HELPER METHODS FROM PREVIOUS RESPONSE BELOW
    # (best_child_rave, run_simulation_rave, get_legal_moves_as_tuples,
    #  board_to_int_matrix, apply_move_int, check_winner_int, safe_fallback_move)
    # --------------------------------------------------------
    def best_child_rave(self, node):
        best_score = -float('inf')
        best_children = []

        for child in node.children.values():
            if child.visits > 0:
                mc_win_rate = child.wins / child.visits
            else:
                mc_win_rate = 0.5

            if child.amaf_visits > 0:
                amaf_win_rate = child.amaf_wins / child.amaf_visits
            else:
                amaf_win_rate = 0.5

            beta = math.sqrt(self.rave_k / (3 * node.visits + self.rave_k))

            # Weighted average
            score = (1 - beta) * mc_win_rate + beta * amaf_win_rate

            if score > best_score:
                best_score = score
                best_children = [child]
            elif score == best_score:
                best_children.append(child)

        return random.choice(best_children)

    def run_simulation_rave(self, matrix, turn_colour):
        empty_spots = []
        size = len(matrix)
        for x in range(size):
            for y in range(size):
                if matrix[x][y] == 0:
                    empty_spots.append((x, y))

        current_val = 1 if turn_colour == Colour.RED else 2
        red_moves = set()
        blue_moves = set()

        while empty_spots:
            idx = random.randrange(len(empty_spots))
            x, y = empty_spots[idx]
            empty_spots[idx] = empty_spots[-1]
            empty_spots.pop()

            matrix[x][y] = current_val
            if current_val == 1:
                red_moves.add((x, y))
            else:
                blue_moves.add((x, y))
            current_val = 3 - current_val

        winner = self.check_winner_int(matrix)
        return winner, red_moves, blue_moves

    def get_legal_moves_as_tuples(self, board):
        moves = []
        for x in range(board.size):
            for y in range(board.size):
                if board.tiles[x][y].colour is None:
                    moves.append((x, y))
        return moves

    def board_to_int_matrix(self, board):
        matrix = [[0] * board.size for _ in range(board.size)]
        for x in range(board.size):
            for y in range(board.size):
                c = board.tiles[x][y].colour
                if c == Colour.RED:
                    matrix[x][y] = 1
                elif c == Colour.BLUE:
                    matrix[x][y] = 2
        return matrix

    def apply_move_int(self, matrix, move_tuple, colour):
        val = 1 if colour == Colour.RED else 2
        matrix[move_tuple[0]][move_tuple[1]] = val

    def check_winner_int(self, matrix):
        size = len(matrix)
        q = []
        visited = set()
        for y in range(size):
            if matrix[0][y] == 1:
                q.append((0, y))
                visited.add((0, y))
        head = 0
        while head < len(q):
            cx, cy = q[head]
            head += 1
            if cx == size - 1: return Colour.RED
            for nx, ny in [(cx - 1, cy), (cx - 1, cy + 1), (cx, cy - 1), (cx, cy + 1), (cx + 1, cy - 1), (cx + 1, cy)]:
                if 0 <= nx < size and 0 <= ny < size and matrix[nx][ny] == 1 and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    q.append((nx, ny))

        q = []
        visited = set()
        for x in range(size):
            if matrix[x][0] == 2:
                q.append((x, 0))
                visited.add((x, 0))
        head = 0
        while head < len(q):
            cx, cy = q[head]
            head += 1
            if cy == size - 1: return Colour.BLUE
            for nx, ny in [(cx - 1, cy), (cx - 1, cy + 1), (cx, cy - 1), (cx, cy + 1), (cx + 1, cy - 1), (cx + 1, cy)]:
                if 0 <= nx < size and 0 <= ny < size and matrix[nx][ny] == 2 and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    q.append((nx, ny))
        return None

    def safe_fallback_move(self, board):
        for x in range(board.size):
            for y in range(board.size):
                if board.tiles[x][y].colour is None:
                    return Move(x, y)
        return Move(0, 0)