import copy
import random
import math
import time
from src.AgentBase import AgentBase
from src.Colour import Colour
from src.Move import Move


class Node:
    def __init__(self, move, parent=None):
        self.move = move  # The move that led to this node
        self.parent = parent
        self.children = {}  # Map move (x,y) -> Node
        self.wins = 0.0
        self.visits = 0
        self.untried_moves = None  # Will be populated on expansion
        self.player_just_moved = None  # Colour of player who made self.move


class MyAgent(AgentBase):
    def __init__(self, colour: Colour):
        super().__init__(colour)
        self.root = None
        self.time_limit = 9.0  # Seconds allowed per turn (safety buffer for 5m total)
        # Assuming 11x11 board for static optimizations
        self.size = 11

    def make_move(self, turn, board, opp_move):
        # 1. Update the tree with opponent's move (Tree Reuse)
        if self.root is not None and opp_move is not None:
            if not opp_move.is_swap():
                # Normal move reuse
                opp_move_key = (opp_move.x, opp_move.y)
                if opp_move_key in self.root.children:
                    self.root = self.root.children[opp_move_key]
                    self.root.parent = None
                else:
                    self.root = None
            else:
                # If opponent swapped, our tree is invalid because colours changed
                self.root = None
        else:
            self.root = None

        # 2. Initialize root if needed
        if self.root is None:
            self.root = Node(None)
            self.root.untried_moves = self.get_legal_moves_as_tuples(board)
            self.root.player_just_moved = Colour.opposite(self.colour)

        # 3. MCTS Loop (Time-based)
        start_time = time.time()
        iterations = 0

        # We keep searching until we hit the time limit
        while (time.time() - start_time) < self.time_limit:
            # Create a lightweight clone for this iteration
            # We only clone the 'skeleton' of the board for speed
            # 0 = Empty, 1 = RED, 2 = BLUE
            sim_board = self.board_to_int_matrix(board)

            # --- SELECTION & EXPANSION ---
            node = self.root
            state_colour = self.colour  # The colour about to move at root

            # Select until we hit a leaf or terminal state
            while node.untried_moves == [] and node.children:
                node = self.best_child(node)
                # Update the board state as we descend
                self.apply_move_int(sim_board, node.move, node.player_just_moved)
                state_colour = Colour.opposite(state_colour)

            # Expand if possible
            if node.untried_moves:
                move_tuple = random.choice(node.untried_moves)

                # Make the move on our sim board
                self.apply_move_int(sim_board, move_tuple, state_colour)

                child = Node(move_tuple, parent=node)
                child.untried_moves = [m for m in node.untried_moves if m != move_tuple]
                child.player_just_moved = state_colour

                # Remove from parent's untried to avoid re-expanding
                node.untried_moves.remove(move_tuple)
                node.children[move_tuple] = child
                node = child

            # --- SIMULATION (ROLLOUT) ---
            # Play random moves until the end on the int-matrix board
            # Determine who moves next in the simulation
            sim_turn_colour = Colour.opposite(node.player_just_moved)
            winner_colour = self.run_lightweight_simulation(sim_board, sim_turn_colour)

            # --- BACKPROPAGATION ---
            while node is not None:
                node.visits += 1
                if node.player_just_moved == winner_colour:
                    node.wins += 1
                node = node.parent

            iterations += 1

        # 4. Select best move
        if not self.root.children:
            return self.safe_fallback_move(board)

        # Robust Child: Select the child with the most visits (not highest win rate)
        best_move_tuple = max(self.root.children, key=lambda k: self.root.children[k].visits)
        best_node = self.root.children[best_move_tuple]

        # 5. Advance our root for the next turn
        self.root = best_node
        self.root.parent = None

        print(f"MCTS Iterations: {iterations}, Win Rate: {best_node.wins / best_node.visits:.2f}")
        return Move(best_move_tuple[0], best_move_tuple[1])

    # --- HELPER METHODS ---

    def best_child(self, node, exploration_constant=1.41):
        # UCB1 Selection
        best_score = -float('inf')
        best_moves = []

        for child in node.children.values():
            if child.visits == 0:
                return child

            exploit = child.wins / child.visits
            explore = exploration_constant * math.sqrt(math.log(node.visits) / child.visits)
            score = exploit + explore

            if score > best_score:
                best_score = score
                best_moves = [child]
            elif score == best_score:
                best_moves.append(child)

        return random.choice(best_moves)

    def get_legal_moves_as_tuples(self, board):
        moves = []
        for x in range(board.size):
            for y in range(board.size):
                if board.tiles[x][y].colour is None:
                    moves.append((x, y))
        return moves

    def board_to_int_matrix(self, board):
        # 0: Empty, 1: RED, 2: BLUE
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

    def run_lightweight_simulation(self, matrix, turn_colour):
        # Get empty spots efficiently
        empty_spots = []
        size = len(matrix)
        for x in range(size):
            for y in range(size):
                if matrix[x][y] == 0:
                    empty_spots.append((x, y))

        current_val = 1 if turn_colour == Colour.RED else 2

        # Randomly fill the rest of the board
        # "Swap and Pop" is faster than normal shuffling
        while empty_spots:
            idx = random.randrange(len(empty_spots))
            # Pick a random move
            x, y = empty_spots[idx]

            # Efficient remove
            empty_spots[idx] = empty_spots[-1]
            empty_spots.pop()

            matrix[x][y] = current_val
            current_val = 3 - current_val  # Switch between 1 and 2

        # Check winner on full board
        return self.check_winner_int(matrix)

    def check_winner_int(self, matrix):
        # Union-Find or BFS to detect win on the matrix
        # 1 = RED (Top-Bottom), 2 = BLUE (Left-Right)
        size = len(matrix)

        # Check RED (1) - Top row to Bottom row
        # BFS
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
            if cx == size - 1:
                return Colour.RED

            # Hex Neighbors
            neighbors = [
                (cx - 1, cy), (cx - 1, cy + 1),
                (cx, cy - 1), (cx, cy + 1),
                (cx + 1, cy - 1), (cx + 1, cy)
            ]
            for nx, ny in neighbors:
                if 0 <= nx < size and 0 <= ny < size:
                    if matrix[nx][ny] == 1 and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        q.append((nx, ny))

        # Check BLUE (2) - Left col to Right col
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
            if cy == size - 1:
                return Colour.BLUE

            neighbors = [
                (cx - 1, cy), (cx - 1, cy + 1),
                (cx, cy - 1), (cx, cy + 1),
                (cx + 1, cy - 1), (cx + 1, cy)
            ]
            for nx, ny in neighbors:
                if 0 <= nx < size and 0 <= ny < size:
                    if matrix[nx][ny] == 2 and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        q.append((nx, ny))

        return None

    def safe_fallback_move(self, board):
        # If something goes wrong, pick the first available tile
        for x in range(board.size):
            for y in range(board.size):
                if board.tiles[x][y].colour is None:
                    return Move(x, y)
        return Move(0, 0)