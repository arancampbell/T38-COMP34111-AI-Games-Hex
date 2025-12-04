import random
import math
import time
from src.AgentBase import AgentBase
from src.Colour import Colour
from src.Move import Move


class Node:
    __slots__ = ('move', 'parent', 'children', 'wins', 'visits',
                 'amaf_wins', 'amaf_visits', 'untried_moves', 'player_just_moved')

    def __init__(self, move, parent=None):
        self.move = move
        self.parent = parent
        self.children = {}
        self.wins = 0.0
        self.visits = 0
        # RAVE (AMAF) Statistics
        self.amaf_wins = 0.0
        self.amaf_visits = 0
        self.untried_moves = None
        self.player_just_moved = None


class My3rdAgent(AgentBase):
    def __init__(self, colour: Colour):
        super().__init__(colour)
        self.root = None
        self.time_limit = 9.0
        self.size = 11
        self.total_tiles = 121

        # --- Precompute Neighbors (1D Graph) ---
        self.neighbors = [[] for _ in range(self.total_tiles)]
        self.red_starts = []
        self.blue_starts = []

        for x in range(self.size):
            for y in range(self.size):
                idx = x * self.size + y
                if x == 0: self.red_starts.append(idx)
                if y == 0: self.blue_starts.append(idx)

                potential = [
                    (x - 1, y), (x - 1, y + 1),
                    (x, y - 1), (x, y + 1),
                    (x + 1, y - 1), (x + 1, y)
                ]
                for nx, ny in potential:
                    if 0 <= nx < self.size and 0 <= ny < self.size:
                        n_idx = nx * self.size + ny
                        self.neighbors[idx].append(n_idx)

    def make_move(self, turn, board, opp_move):
        # 1. DISABLE TREE REUSE
        # Clearing the tree forces the agent to re-evaluate the board based on
        # the ACTUAL current threat, not what it thought 10 turns ago.
        self.root = None

        if self.root is None:
            self.root = Node(None)
            self.root.untried_moves = self.get_legal_moves_as_tuples(board)
            self.root.player_just_moved = Colour.opposite(self.colour)

        # 2. Defensive Solver (Instant Win/Loss check)
        board_1d = self.board_to_1d(board)
        legal_indices = [i for i, v in enumerate(board_1d) if v == 0]

        my_val = 1 if self.colour == Colour.RED else 2
        opp_val = 3 - my_val

        # Check instant win
        for idx in legal_indices:
            board_1d[idx] = my_val
            if self.check_winner_1d(board_1d) == my_val:
                r, c = divmod(idx, self.size)
                return Move(r, c)
            board_1d[idx] = 0

        # Check instant loss (Must block)
        must_block_indices = []
        for idx in legal_indices:
            board_1d[idx] = opp_val
            if self.check_winner_1d(board_1d) == opp_val:
                must_block_indices.append(idx)
            board_1d[idx] = 0

        if must_block_indices:
            print(f"Solver: Found {len(must_block_indices)} critical threats. Focusing search.")
            target_moves = set((idx // self.size, idx % self.size) for idx in must_block_indices)
            self.root.untried_moves = [m for m in self.root.untried_moves if m in target_moves]

        # 3. RAVE MCTS Loop
        start_time = time.time()
        iterations = 0
        my_colour_code = 1 if self.colour == Colour.RED else 2

        # Get the index of the opponent's last move for the simulation heuristic
        last_opp_idx = -1
        if opp_move and not opp_move.is_swap():
            last_opp_idx = opp_move.x * self.size + opp_move.y

        while (time.time() - start_time) < self.time_limit:
            sim_board = board_1d[:]
            node = self.root
            current_player_code = my_colour_code

            moves_in_tree = set()

            # --- SELECTION ---
            while node.untried_moves == [] and node.children:
                node = self.best_child_rave(node)
                moves_in_tree.add(node.move)
                mx, my = node.move
                idx = mx * self.size + my
                sim_board[idx] = 1 if node.player_just_moved == Colour.RED else 2
                current_player_code = 3 - current_player_code

            # --- EXPANSION ---
            if node.untried_moves:
                idx_in_list = random.randrange(len(node.untried_moves))
                move_tuple = node.untried_moves[idx_in_list]
                node.untried_moves[idx_in_list] = node.untried_moves[-1]
                node.untried_moves.pop()

                mx, my = move_tuple
                idx = mx * self.size + my

                val = 1 if current_player_code == 1 else 2
                sim_board[idx] = val

                child = Node(move_tuple, parent=node)
                child.untried_moves = node.untried_moves[:]
                child.player_just_moved = self.colour if current_player_code == 1 else Colour.opposite(self.colour)

                node.children[move_tuple] = child
                node = child
                moves_in_tree.add(move_tuple)
                current_player_code = 3 - current_player_code

                # Update last move for simulation
                last_node_idx = idx
            else:
                last_node_idx = last_opp_idx

            # --- SIMULATION (Heuristic) ---
            # We pass the last move index to play locally
            winner_code, moves_in_sim = self.run_simulation_smart(sim_board, current_player_code, last_node_idx)

            # --- BACKPROPAGATION ---
            winner_colour = Colour.RED if winner_code == 1 else Colour.BLUE
            all_moves = moves_in_tree.union(moves_in_sim)

            while node is not None:
                node.visits += 1
                if node.player_just_moved == winner_colour:
                    node.wins += 1

                for move, child in node.children.items():
                    if move in all_moves:
                        child.amaf_visits += 1
                        if child.player_just_moved == winner_colour:
                            child.amaf_wins += 1

                node = node.parent

            iterations += 1

        # 4. Final Selection
        if not self.root.children:
            return self.safe_fallback_move(board)

        # Robust Child Selection
        best_move_tuple = max(self.root.children, key=lambda k: self.root.children[k].visits)
        best_node = self.root.children[best_move_tuple]

        win_rate = best_node.wins / best_node.visits if best_node.visits > 0 else 0.0
        print(f"Smart-RAVE MCTS: {iterations} iter, Rate: {win_rate:.2f}")
        return Move(best_move_tuple[0], best_move_tuple[1])

    # --- RAVE HELPERS ---

    def best_child_rave(self, node):
        # Lower 'k' trusts the tree stats sooner (less hallucination)
        k = 50
        best_score = -float('inf')
        best_moves = []

        for child in node.children.values():
            if child.visits == 0:
                beta = 1.0
                uct_score = 0.5
            else:
                beta = k / (k + child.visits)
                uct_score = child.wins / child.visits

            if child.amaf_visits > 0:
                amaf_score = child.amaf_wins / child.amaf_visits
            else:
                amaf_score = 0.5

            if child.visits > 0:
                explore = 0.4 * math.sqrt(math.log(node.visits) / child.visits)
            else:
                explore = 1.0

            score = (1 - beta) * uct_score + beta * amaf_score + explore

            if score > best_score:
                best_score = score
                best_moves = [child]
            elif score == best_score:
                best_moves.append(child)

        return random.choice(best_moves)

    def run_simulation_smart(self, board, turn_code, last_idx):
        """
        Smart Simulation: Tries to play a neighbor of the last move
        with 50% probability. This simulates local fights/walls.
        """
        empty_indices = [i for i, x in enumerate(board) if x == 0]
        # Optimize: Map indices for O(1) removal?
        # For Python, shuffle is still fastest, but we will tweak the ORDER.

        # If we have a last move, swap a neighbor to the front
        if last_idx != -1 and 0 <= last_idx < 121:
            # Pick a random neighbor of the last move
            potential_neighbors = self.neighbors[last_idx]
            # Try to find an empty neighbor
            for n_idx in potential_neighbors:
                if board[n_idx] == 0:
                    # Move this neighbor to the front of empty_indices logic?
                    # Since empty_indices is a list, finding n_idx is O(N).
                    # This is too slow.
                    # Instead, we just PLAY it immediately if valid.
                    board[n_idx] = turn_code
                    # Remove from empty_indices (slow) or just proceed.
                    # To keep it fast:
                    # We just play random moves, but we can't easily do the heuristic
                    # without slowing down the iterations.

                    # Compromise: Pure Random is actually better IF tree is reset.
                    break

        # Standard Shuffle (Fastest in Python)
        random.shuffle(empty_indices)

        current = turn_code
        moves_in_sim = set()

        for idx in empty_indices:
            if board[idx] == 0:  # Check if not filled by heuristic above
                board[idx] = current
                r, c = divmod(idx, self.size)
                moves_in_sim.add((r, c))
                current = 3 - current

        return self.check_winner_1d(board), moves_in_sim

    def check_winner_1d(self, board):
        # Optimized DFS
        stack = [i for i in self.red_starts if board[i] == 1]
        visited = [False] * 121
        for i in stack: visited[i] = True
        while stack:
            curr = stack.pop()
            if curr >= 110: return 1
            for n_idx in self.neighbors[curr]:
                if board[n_idx] == 1 and not visited[n_idx]:
                    visited[n_idx] = True
                    stack.append(n_idx)

        stack = [i for i in self.blue_starts if board[i] == 2]
        visited = [False] * 121
        for i in stack: visited[i] = True
        while stack:
            curr = stack.pop()
            if curr % 11 == 10: return 2
            for n_idx in self.neighbors[curr]:
                if board[n_idx] == 2 and not visited[n_idx]:
                    visited[n_idx] = True
                    stack.append(n_idx)
        return 0

    def get_legal_moves_as_tuples(self, board):
        moves = []
        for x in range(board.size):
            for y in range(board.size):
                if board.tiles[x][y].colour is None:
                    moves.append((x, y))
        return moves

    def board_to_1d(self, board):
        flat = [0] * (board.size * board.size)
        for x in range(board.size):
            for y in range(board.size):
                c = board.tiles[x][y].colour
                idx = x * board.size + y
                if c == Colour.RED:
                    flat[idx] = 1
                elif c == Colour.BLUE:
                    flat[idx] = 2
        return flat

    def safe_fallback_move(self, board):
        for x in range(board.size):
            for y in range(board.size):
                if board.tiles[x][y].colour is None:
                    return Move(x, y)
        return Move(0, 0)