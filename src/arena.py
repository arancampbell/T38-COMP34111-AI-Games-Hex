import sys
import time
import importlib
import pandas as pd
from collections import defaultdict
from src.Colour import Colour
from src.Game import Game
from src.Player import Player
from src.AgentBase import AgentBase

# --- CONFIGURATION ---
TIME_CONSTRAINTS = [1, 3, 5]  # Minutes
GAMES_PER_SET = 10  # Games per time limit (5 as Red, 5 as Blue)


class StatsWrapper(AgentBase):
    """Wraps an agent to intercept calls and track win rates."""

    def __init__(self, agent_instance, name, tracker_dict):
        self.agent = agent_instance
        self.name = name
        self.tracker = tracker_dict
        self._colour = agent_instance.colour

    @property
    def colour(self):
        return self.agent.colour

    @colour.setter
    def colour(self, val):
        self.agent.colour = val
        self._colour = val

    def make_move(self, turn, board, opp_move):
        # Delegate move to real agent
        move = self.agent.make_move(turn, board, opp_move)

        # Track Win Rate if available (every 10 turns)
        if turn % 10 == 0:
            # Check for standard attribute names for win rate
            wr = getattr(self.agent, 'last_win_rate', None)
            if wr is None: wr = getattr(self.agent, 'win_rate', None)

            if wr is not None:
                if turn not in self.tracker:
                    self.tracker[turn] = {}
                self.tracker[turn][self.name] = wr

        return move


def load_agent(path_str, class_name, colour):
    """Dynamically loads an agent class from a string path."""
    try:
        module_path = path_str.replace("/", ".").replace("\\", ".")
        if module_path.endswith(".py"):
            module_path = module_path[:-3]

        module = importlib.import_module(module_path)
        agent_class = getattr(module, class_name)
        return agent_class(colour)
    except Exception as e:
        print(f"Failed to load agent {class_name} from {path_str}: {e}")
        sys.exit(1)


def run_tournament(agent1_info, agent2_info):
    """
    agent1_info: ("path.to.Agent", "ClassName", "Agent Name")
    """
    results = {}

    for mins in TIME_CONSTRAINTS:
        print(f"\n{'=' * 60}")
        print(f"STARTING TOURNAMENT: {mins} MINUTE LIMIT")
        print(f"{'=' * 60}")

        # 1. Hack the Global Time Limit
        # Game.MAXIMUM_TIME is in nanoseconds
        # We set it slightly lower than the 'Arena' limit to ensure Game handles the timeout logic
        # strictly if the agents fail to manage their own time.
        Game.MAXIMUM_TIME = (mins * 60) * 10 ** 9

        wins = defaultdict(int)
        win_probs = {}  # { turn_num: {agentA: [0.5, 0.6], agentB: [...]} }

        # Run games
        for i in range(GAMES_PER_SET):
            # Swap colors every game to ensure fairness
            # Even games: A=Red, B=Blue
            # Odd games: B=Red, A=Blue
            if i % 2 == 0:
                p1_def, p2_def = agent1_info, agent2_info
            else:
                p1_def, p2_def = agent2_info, agent1_info

            # Load Fresh Agents
            raw_agent1 = load_agent(p1_def[0], p1_def[1], Colour.RED)
            raw_agent2 = load_agent(p2_def[0], p2_def[1], Colour.BLUE)

            # Inject Time Limit into Agents if they support it (Dynamic Budgeting)
            # Assuming attributes like 'GAME_TIME_LIMIT' or 'time_limit'
            limit_sec = mins * 60
            if hasattr(raw_agent1, 'GAME_TIME_LIMIT'): raw_agent1.GAME_TIME_LIMIT = limit_sec - 2
            if hasattr(raw_agent2, 'GAME_TIME_LIMIT'): raw_agent2.GAME_TIME_LIMIT = limit_sec - 2

            # Wrap Agents for Stats
            game_stats = {}
            wrapped_a1 = StatsWrapper(raw_agent1, p1_def[2], game_stats)
            wrapped_a2 = StatsWrapper(raw_agent2, p2_def[2], game_stats)

            player1 = Player(p1_def[2], wrapped_a1)
            player2 = Player(p2_def[2], wrapped_a2)

            print(f"  Game {i + 1}/{GAMES_PER_SET}: {player1.name} (Red) vs {player2.name} (Blue)...", end="",
                  flush=True)

            # Run Game (Silent mode to reduce clutter)
            game = Game(player1, player2, verbose=False, silent=True)
            result = game.run()

            winner_name = result['winner']
            wins[winner_name] += 1
            print(f" Winner: {winner_name}")

            # Aggregate Win Probs
            for turn, rates in game_stats.items():
                if turn not in win_probs: win_probs[turn] = defaultdict(list)
                for ag_name, rate in rates.items():
                    win_probs[turn][ag_name].append(rate)

        # Store Results for this Time Constraint
        results[mins] = {
            "wins": dict(wins),
            "probs": win_probs
        }

    return results


def print_report(results, a1_name, a2_name):
    print("\n\n" + "#" * 60)
    print("FINAL ARENA REPORT")
    print("#" * 60)

    for mins, data in results.items():
        print(f"\n--- TIME LIMIT: {mins} MINUTES ---")

        # 1. Win Rates
        wins = data['wins']
        a1_wins = wins.get(a1_name, 0)
        a2_wins = wins.get(a2_name, 0)
        total = a1_wins + a2_wins
        if total == 0: total = 1  # Avoid div/0

        print(f"MATCH SCORE: {a1_name} {a1_wins} - {a2_wins} {a2_name}")
        print(f"WIN RATE ({a1_name}): {(a1_wins / total) * 100:.1f}%")

        # 2. Confidence Evolution (Win Probabilities)
        print("\nAVERAGE CONFIDENCE (Self-Estimated Win %):")
        probs = data['probs']
        sorted_turns = sorted(probs.keys())

        # Create a clean table row
        print(f"{'Turn':<6} | {a1_name:<15} | {a2_name:<15}")
        print("-" * 42)

        for turn in sorted_turns:
            rates = probs[turn]

            # Calculate average confidence for this turn across all games
            a1_vals = rates.get(a1_name, [])
            a2_vals = rates.get(a2_name, [])

            a1_avg = f"{sum(a1_vals) / len(a1_vals):.2f}" if a1_vals else "-"
            a2_avg = f"{sum(a2_vals) / len(a2_vals):.2f}" if a2_vals else "-"

            print(f"{turn:<6} | {a1_avg:<15} | {a2_avg:<15}")


if __name__ == "__main__":
    # EDIT THESE TO POINT TO YOUR AGENTS
    # Format: (FilePath (dot notation), ClassName, DisplayName)

    # Example: My Optimized Agent
    AGENT_A = ("agents.Group38.BestAgent", "BestAgent", "BestAgent")

    # Example: The "Non-Optimized" Agent
    AGENT_B = ("agents.Group38.My2ndAgent", "My2ndAgent", "My2ndAgent")

    # Run
    results = run_tournament(AGENT_A, AGENT_B)
    print_report(results, AGENT_A[2], AGENT_B[2])