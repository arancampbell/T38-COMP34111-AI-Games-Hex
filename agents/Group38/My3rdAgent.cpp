#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>
#include <limits>
#include <cstring>
#include <string>
#include <sstream>

// --- CONSTANTS ---
const int SIZE = 11;
const int TOTAL_TILES = 121;
const double RAVE_K = 50.0;
const double EXPLORATION = 0.4;
const double TIME_BUFFER = 0.5;
const double TOTAL_GAME_TIME = 300.0; // 5 Minutes

enum Colour { RED = 1, BLUE = 2, EMPTY = 0 };

struct Move {
    int x, y;
    Move(int _x = -1, int _y = -1) : x(_x), y(_y) {}
};

// --- NODE STRUCTURE ---
struct Node {
    int move_idx; // 0-120
    Node* parent;
    std::vector<Node*> children;
    std::vector<int> untried_moves;
    int player_just_moved; // RED or BLUE

    double wins;
    int visits;
    double amaf_wins;
    int amaf_visits;

    Node(int move, Node* par, const std::vector<int>& legal_moves, int player)
        : move_idx(move), parent(par), player_just_moved(player),
          wins(0.0), visits(0), amaf_wins(0.0), amaf_visits(0) {
        untried_moves = legal_moves;
    }

    ~Node() {
        for (Node* child : children) delete child;
    }
};

class My3rdAgent {
private:
    int my_colour;
    int opp_colour;
    double total_time_used;
    std::vector<int> neighbors[TOTAL_TILES];
    std::vector<int> red_starts, blue_starts;
    std::mt19937 rng;

public:
    My3rdAgent(int colour) : my_colour(colour), total_time_used(0.0) {
        opp_colour = (my_colour == RED) ? BLUE : RED;

        // Seed RNG
        std::random_device rd;
        rng.seed(rd());

        // Precompute Neighbors
        for (int x = 0; x < SIZE; ++x) {
            for (int y = 0; y < SIZE; ++y) {
                int idx = x * SIZE + y;
                if (x == 0) red_starts.push_back(idx);
                if (y == 0) blue_starts.push_back(idx);

                int potential[6][2] = {
                    {x - 1, y}, {x - 1, y + 1},
                    {x, y - 1}, {x, y + 1},
                    {x + 1, y - 1}, {x + 1, y}
                };

                for (auto& p : potential) {
                    if (p[0] >= 0 && p[0] < SIZE && p[1] >= 0 && p[1] < SIZE) {
                        neighbors[idx].push_back(p[0] * SIZE + p[1]);
                    }
                }
            }
        }
    }

    // --- MAIN MOVE FUNCTION ---
    Move make_move(int turn, const std::vector<int>& board, Move opp_move) {
        auto start_time = std::chrono::high_resolution_clock::now();

        // 0. OPENING BOOK
        if (turn == 1 && my_colour == RED) return Move(5, 5);
        if (turn == 2 && my_colour == BLUE && !is_swap(opp_move)) {
            if (opp_move.x >= 3 && opp_move.x <= 7 && opp_move.y >= 3 && opp_move.y <= 7) {
                return Move(-1, -1); // Swap
            }
        }

        // 1. TIME MANAGEMENT
        double time_limit = get_time_budget(board);

        // 2. DEFENSIVE SOLVER
        std::vector<int> legal_indices;
        for(int i=0; i<TOTAL_TILES; ++i) if(board[i] == EMPTY) legal_indices.push_back(i);

        // Check Instant Win
        std::vector<int> temp_board = board;
        for (int idx : legal_indices) {
            temp_board[idx] = my_colour;
            if (check_winner(temp_board) == my_colour) {
                update_time(start_time);
                return idx_to_move(idx);
            }
            temp_board[idx] = EMPTY;
        }

        // Check Instant Loss (Must Block)
        std::vector<int> forced_moves;
        for (int idx : legal_indices) {
            temp_board[idx] = opp_colour;
            if (check_winner(temp_board) == opp_colour) {
                forced_moves.push_back(idx);
            }
            temp_board[idx] = EMPTY;
        }

        // 3. MCTS SETUP
        Node* root = new Node(-1, nullptr, legal_indices, opp_colour);

        // If forced, prune tree
        if (!forced_moves.empty()) {
            root->untried_moves = forced_moves;
        }

        int iterations = 0;
        int last_opp_idx = (is_swap(opp_move)) ? -1 : (opp_move.x * SIZE + opp_move.y);

        // 4. MCTS LOOP
        while (get_elapsed(start_time) < time_limit) {
            Node* node = root;
            std::vector<int> sim_board = board;
            std::vector<int> moves_in_tree;
            int current_player = my_colour;

            // SELECTION
            while (node->untried_moves.empty() && !node->children.empty()) {
                node = best_child_rave(node);
                moves_in_tree.push_back(node->move_idx);
                sim_board[node->move_idx] = (node->player_just_moved == RED) ? RED : BLUE;
                current_player = (current_player == RED) ? BLUE : RED;
            }

            // EXPANSION
            int last_node_idx = last_opp_idx;
            if (!node->untried_moves.empty()) {
                // Swap-and-Pop random selection
                std::uniform_int_distribution<int> dist(0, node->untried_moves.size() - 1);
                int rand_idx = dist(rng);
                int move_idx = node->untried_moves[rand_idx];

                // Remove fast
                node->untried_moves[rand_idx] = node->untried_moves.back();
                node->untried_moves.pop_back();

                sim_board[move_idx] = current_player;

                Node* child = new Node(move_idx, node, node->untried_moves, current_player);
                node->children.push_back(child);

                node = child;
                moves_in_tree.push_back(move_idx);
                last_node_idx = move_idx;
                current_player = (current_player == RED) ? BLUE : RED;
            }

            // SIMULATION (Smart)
            std::vector<int> moves_in_sim;
            int winner = run_simulation_smart(sim_board, current_player, last_node_idx, moves_in_sim);

            // BACKPROPAGATION
            while (node != nullptr) {
                node->visits++;
                if (node->player_just_moved == winner) node->wins++;

                // RAVE Update
                for (Node* child : node->children) {
                    bool found = false;
                    for(int m : moves_in_sim) if(m == child->move_idx) { found=true; break; }
                    if(!found) {
                        for(int m : moves_in_tree) if(m == child->move_idx) { found=true; break; }
                    }

                    if (found) {
                        child->amaf_visits++;
                        if (child->player_just_moved == winner) child->amaf_wins++;
                    }
                }
                node = node->parent;
            }
            iterations++;
        }

        // 5. FINAL SELECTION
        Move best_move(-1, -1);
        int max_visits = -1;
        double win_rate = 0.0;

        if (root->children.empty()) {
            best_move = safe_fallback(board);
        } else {
            Node* best_node = nullptr;
            for (Node* child : root->children) {
                if (child->visits > max_visits) {
                    max_visits = child->visits;
                    best_move = idx_to_move(child->move_idx);
                    best_node = child;
                }
            }
            // Calculate win rate for the log
            if (best_node != nullptr && best_node->visits > 0) {
                win_rate = (double)best_node->wins / best_node->visits;
            }
        }

        // Calculate timing
        double current_move_time = get_elapsed(start_time);
        update_time(start_time); // updates total_time_used

        // --- PRINT STATS TO STDERR ---
        std::cerr << "\n-------C++ Agent: "
                  << iterations << " iterations ("
                  << std::fixed << std::setprecision(2) << current_move_time << "s/"
                  << total_time_used << "s), Win rate: "
                  << win_rate << "-------\n" << std::endl;

        delete root; // Clean up memory
        return best_move;
    }

private:
    double get_elapsed(std::chrono::high_resolution_clock::time_point start) {
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = now - start;
        return diff.count();
    }

    void update_time(std::chrono::high_resolution_clock::time_point start) {
        total_time_used += get_elapsed(start);
    }

    double get_time_budget(const std::vector<int>& board) {
        double remaining = (TOTAL_GAME_TIME - TIME_BUFFER) - total_time_used;
        int empty = 0;
        for(int x : board) if(x == EMPTY) empty++;

        double turns_left = std::max(1.0, empty / 2.0);
        double budget = remaining / turns_left;

        if (empty > 20 && empty < 80) budget = std::min(budget * 1.4, 7.0);
        else budget = std::min(budget, 4.0);

        return std::max(budget, 0.2);
    }

    Node* best_child_rave(Node* node) {
        double best_score = -1e9;
        std::vector<Node*> best_nodes;
        double log_visits = std::log(node->visits);

        for (Node* child : node->children) {
            double beta, uct, amaf;
            if (child->visits == 0) {
                beta = 1.0;
                uct = 0.5;
            } else {
                beta = RAVE_K / (RAVE_K + child->visits);
                uct = child->wins / child->visits;
            }

            if (child->amaf_visits > 0) amaf = child->amaf_wins / child->amaf_visits;
            else amaf = 0.5;

            double explore = (child->visits > 0) ?
                EXPLORATION * std::sqrt(log_visits / child->visits) : 1.0;

            double score = (1.0 - beta) * uct + beta * amaf + explore;

            if (score > best_score) {
                best_score = score;
                best_nodes.clear();
                best_nodes.push_back(child);
            } else if (score == best_score) {
                best_nodes.push_back(child);
            }
        }
        std::uniform_int_distribution<int> dist(0, best_nodes.size() - 1);
        return best_nodes[dist(rng)];
    }

    int run_simulation_smart(std::vector<int>& board, int turn, int last_idx, std::vector<int>& moves_made) {
        std::vector<int> empty;
        for(int i=0; i<TOTAL_TILES; ++i) if(board[i] == EMPTY) empty.push_back(i);

        // Smart Heuristic: Check neighbors of last move
        if (last_idx != -1) {
            for (int n_idx : neighbors[last_idx]) {
                if (board[n_idx] == EMPTY) {
                    board[n_idx] = turn;
                    moves_made.push_back(n_idx);
                    turn = (turn == RED) ? BLUE : RED;
                    // Remove from empty list (slow O(N), but acceptable for sim accuracy)
                    empty.erase(std::remove(empty.begin(), empty.end(), n_idx), empty.end());
                    break;
                }
            }
        }

        std::shuffle(empty.begin(), empty.end(), rng);

        for (int idx : empty) {
            board[idx] = turn;
            moves_made.push_back(idx);
            turn = (turn == RED) ? BLUE : RED;
        }

        return check_winner(board);
    }

    int check_winner(const std::vector<int>& board) {
        // BFS/DFS for Winner Check
        // RED (Top-Down)
        std::vector<int> stack;
        bool visited[TOTAL_TILES] = {false};

        for (int start : red_starts) if (board[start] == RED) { stack.push_back(start); visited[start] = true; }

        while (!stack.empty()) {
            int curr = stack.back(); stack.pop_back();
            if (curr >= 110) return RED;
            for (int n : neighbors[curr]) {
                if (board[n] == RED && !visited[n]) {
                    visited[n] = true;
                    stack.push_back(n);
                }
            }
        }

        // BLUE (Left-Right)
        stack.clear();
        std::fill(std::begin(visited), std::end(visited), false);
        for (int start : blue_starts) if (board[start] == BLUE) { stack.push_back(start); visited[start] = true; }

        while (!stack.empty()) {
            int curr = stack.back(); stack.pop_back();
            if (curr % SIZE == 10) return BLUE;
            for (int n : neighbors[curr]) {
                if (board[n] == BLUE && !visited[n]) {
                    visited[n] = true;
                    stack.push_back(n);
                }
            }
        }
        return 0; // Draw/None
    }

    bool is_swap(Move m) { return m.x == -1 && m.y == -1; }
    Move idx_to_move(int idx) { return Move(idx / SIZE, idx % SIZE); }
    Move safe_fallback(const std::vector<int>& board) {
        for(int i=0; i<TOTAL_TILES; ++i) if(board[i] == EMPTY) return idx_to_move(i);
        return Move(0,0);
    }
};

// --- HELPER FUNCTIONS FOR PROTOCOL ---

std::vector<std::string> split(const std::string &s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

std::vector<int> parse_board_string(const std::string& board_str, int size) {
    // Protocol: "000,000,000" -> 1D vector
    std::vector<int> board(size * size, 0);
    int row = 0;
    int col = 0;

    for (char c : board_str) {
        if (c == ',') {
            row++;
            col = 0;
        } else {
            int idx = row * size + col;
            if (c == 'R') board[idx] = RED;
            else if (c == 'B') board[idx] = BLUE;
            else board[idx] = EMPTY; // '0'
            col++;
        }
    }
    return board;
}

Move parse_move_string(const std::string& move_str) {
    if (move_str.empty()) return Move(-1, -1);
    std::vector<std::string> parts = split(move_str, ',');
    if (parts.size() != 2) return Move(-1, -1);
    return Move(std::stoi(parts[0]), std::stoi(parts[1]));
}

// --- MAIN INTERFACE ---

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Error: Missing arguments. Usage: ./Agent [R/B] [Size]" << std::endl;
        return 1;
    }

    char colour_char = argv[1][0];
    int size = (argc >= 3) ? std::stoi(argv[2]) : 11;

    int my_colour = (colour_char == 'R') ? RED : BLUE;
    My3rdAgent agent(my_colour);

    // 2. Game Loop
    std::string line;
    while (std::getline(std::cin, line)) {
        // Protocol: COMMAND;MOVE;BOARD;TURN;
        std::vector<std::string> parts = split(line, ';');

        if (parts.size() < 4) continue;

        std::string command = parts[0];
        std::string move_str = parts[1];
        std::string board_str = parts[2];
        std::string turn_str = parts[3];

        int turn = std::stoi(turn_str);

        // Parse Opponent Move
        Move opp_move(-1, -1);

        if (command == "SWAP") {
            opp_move = Move(-1, -1);
        } else if (!move_str.empty()) {
            opp_move = parse_move_string(move_str);
        }

        // Parse Board
        std::vector<int> board = parse_board_string(board_str, size);

        // Ask Agent for Move
        Move best_move = agent.make_move(turn, board, opp_move);

        // 3. Send Response
        std::cout << best_move.x << "," << best_move.y << std::endl;
    }

    return 0;
}