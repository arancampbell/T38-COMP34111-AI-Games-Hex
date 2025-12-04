import copy
import random
import math

from src.AgentBase import AgentBase
from src.Colour import Colour
from src.Move import Move


class Node:
    def __init__(self, board_state, player_to_move, parent=None, move_from_parent=None, untried_moves=None):
        self.state = board_state
        self.player_to_move = player_to_move
        self.parent = parent
        self.children = []
        self.move_from_parent = move_from_parent
        self.visits = 0
        self.wins = 0.0
        if untried_moves is None:
            self.untried_moves = []
        else:
            self.untried_moves = list(untried_moves)


class MyOriginalAgent(AgentBase):
    def __init__(self, colour: Colour):
        super().__init__(colour)

        self.total_time_used = 0
        self.board_size = None

        self.has_swapped = False
        self.move_number = 0

    def make_move(self, turn, board, opp_move):
        self.move_number += 1

        root_board_state = copy.deepcopy(board)
        root_player = self.colour

        if self.state_is_terminal(root_board_state):
            root_untried_moves = []
        else:
            root_untried_moves = self.get_legal_moves(root_board_state)

        root = Node(
            board_state=root_board_state,
            player_to_move=root_player,
            parent=None,
            move_from_parent=None,
            untried_moves=root_untried_moves
        )

        NUM_ITERATIONS = 500

        for _ in range(NUM_ITERATIONS):
            leaf = self.tree_policy(root)
            winner = self.default_policy(leaf.state, leaf.player_to_move)

            if winner is None:
                continue

            self.backup(leaf, winner, root_player)

        if len(root.children) == 0:
            return self.safe_fallback_move(board)
        else:
            best_child = self.best_child(root, exploration_constant=0.0)
            chosen_move = best_child.move_from_parent

            if chosen_move is None:
                return self.safe_fallback_move(board)

            if chosen_move.x < 0 or chosen_move.y < 0 or \
                    chosen_move.x >= board.size or chosen_move.y >= board.size:
                return self.safe_fallback_move(board)

            if board.tiles[chosen_move.x][chosen_move.y].colour is not None:
                return self.safe_fallback_move(board)

            return chosen_move

    def get_legal_moves(self, board_state):
        moves = []
        for x in range(board_state.size):
            for y in range(board_state.size):
                if board_state.tiles[x][y].colour is None:
                    moves.append(Move(x, y))
        return moves

    def board_state_after_move(self, board_state, move, player_colour):
        board_state.set_tile_colour(move.x, move.y, player_colour)
        return board_state

    def safe_fallback_move(self, board):
        legal_moves = self.get_legal_moves(board)
        if len(legal_moves) != 0:
            return legal_moves[0]
        else:
            return Move(0, 0)

    def state_is_terminal(self, board_state):
        if board_state.has_ended(Colour.RED):
            return True
        if board_state.has_ended(Colour.BLUE):
            return True
        return False

    def get_winner(self, board_state):
        if board_state.has_ended(Colour.RED) or board_state.has_ended(Colour.BLUE):
            return board_state.get_winner()
        return None

    def default_policy(self, board_state, player_to_move):
        temp_board_state = copy.deepcopy(board_state)
        current_player = player_to_move

        while not self.state_is_terminal(temp_board_state):
            legal_moves = self.get_legal_moves(temp_board_state)
            if len(legal_moves) == 0:
                break

            chosen_move = random.choice(legal_moves)
            temp_board_state = self.board_state_after_move(temp_board_state, chosen_move, current_player)
            current_player = Colour.opposite(current_player)

        winner = self.get_winner(temp_board_state)
        return winner

    def best_child(self, node, exploration_constant):
        best_child = None
        best_value = -math.inf

        for child in node.children:
            exploitation = child.wins / child.visits
            exploration = exploration_constant * math.sqrt(math.log(node.visits) / child.visits)

            score = exploration + exploitation
            if score > best_value:
                best_value = score
                best_child = child

        return best_child

    def expand(self, node):
        untried_move = node.untried_moves.pop()

        new_board_state = self.board_state_after_move(
            copy.deepcopy(node.state),
            untried_move,
            node.player_to_move
        )

        next_player = Colour.opposite(node.player_to_move)

        if self.state_is_terminal(new_board_state):
            child_untried_moves = []
        else:
            child_untried_moves = self.get_legal_moves(new_board_state)

        child = Node(
            board_state=new_board_state,
            player_to_move=next_player,
            parent=node,
            move_from_parent=untried_move,
            untried_moves=child_untried_moves
        )

        node.children.append(child)
        return child

    def tree_policy(self, node):
        current_node = node
        while not self.state_is_terminal(current_node.state):
            if len(current_node.untried_moves) != 0:
                return self.expand(current_node)
            else:
                current_node = self.best_child(current_node, exploration_constant=1.4)

        return current_node

    def backup(self, node, winner, root_player):
        current_node = node

        while current_node is not None:
            current_node.visits += 1
            if winner == root_player:
                current_node.wins += 1
            current_node = current_node.parent
