#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <string>
#include <sstream>

// --- CONSTANTS ---
const int SIZE = 11;
const int TOTAL_TILES = 121;
const double RAVE_K = 50.0;
const double EXPLORATION = 0.4;
const double TIME_BUFFER = 0.5;
const double TOTAL_GAME_TIME = 300.0;
const int MAX_NODES = 800000; // Pre-allocated node pool size

enum Colour { EMPTY = 0, RED = 1, BLUE = 2 };

struct Move {
    int x, y;
    Move(int _x = -1, int _y = -1) : x(_x), y(_y) {}
};

// --- FAST RANDOM (Xorshift) ---
struct FastRandom {
    uint64_t s[2];
    FastRandom() {
        s[0] = 0xDEADBEEF;
        s[1] = 0xCAFEBABE;
    }
    // Very fast 0..n-1 generator
    inline int next_int(int n) {
        uint64_t x = s[0];
        uint64_t const y = s[1];
        s[0] = y;
        x ^= x << 23;
        s[1] = x ^ y ^ (x >> 17) ^ (y >> 26);
        return (s[1] + y) % n;
    }
    // 0.0 to 1.0
    inline double next_double() {
        return next_int(1000000) / 1000000.0;
    }
};

// --- NODE STRUCTURE (Optimized) ---
struct Node {
    int move_idx;
    int parent_idx;       // Index in pool
    int first_child;      // Index in pool (-1 if none)
    int next_sibling;     // Index in pool (-1 if none)
    
    // We store untried moves as a simple count to save memory
    // and generate them on the fly or swap-remove from a list.
    // For simplicity in this optimization, we'll keep a small vector
    // but typically you'd optimize this further.
    std::vector<int> untried_moves; 
    
    int player_just_moved;
    float wins;           // Float is sufficient
    int visits;
    float amaf_wins;
    int amaf_visits;

    void init(int move, int par, const std::vector<int>& legal, int player) {
        move_idx = move;
        parent_idx = par;
        first_child = -1;
        next_sibling = -1;
        untried_moves = legal;
        player_just_moved = player;
        wins = 0; visits = 0;
        amaf_wins = 0; amaf_visits = 0;
    }
};

// Global Node Pool to avoid 'new' overhead
Node node_pool[MAX_NODES];
int pool_ptr = 0;

class BestAgent {
private:
    int my_colour;
    int opp_colour;
    double total_time_used;
    
    // Flattened neighbors for speed [tile][0..5]
    int neighbors[TOTAL_TILES][6];
    int neighbor_count[TOTAL_TILES];
    
    // Flattened bridge partners for heuristic
    // bridge_partners[tile][dir] = response_idx
    // We use -1 to indicate no bridge response
    int bridge_partners[TOTAL_TILES][6]; 

    std::vector<int> red_starts, blue_starts;
    FastRandom rng;

public:
    BestAgent(int colour) : my_colour(colour), total_time_used(0.0) {
        opp_colour = (my_colour == RED) ? BLUE : RED;

        // 1. Precompute Neighbors & Starts
        for (int i = 0; i < TOTAL_TILES; ++i) {
            neighbor_count[i] = 0;
            for(int k=0; k<6; ++k) {
                neighbors[i][k] = -1;
                bridge_partners[i][k] = -1;
            }

            int r = i / SIZE;
            int c = i % SIZE;

            if (r == 0) red_starts.push_back(i);
            if (c == 0) blue_starts.push_back(i);

            int potential[6][2] = {
                {r - 1, c}, {r - 1, c + 1},
                {r, c - 1}, {r, c + 1},
                {r + 1, c - 1}, {r + 1, c}
            };

            for (int k = 0; k < 6; ++k) {
                int nr = potential[k][0];
                int nc = potential[k][1];
                if (nr >= 0 && nr < SIZE && nc >= 0 && nc < SIZE) {
                    neighbors[i][neighbor_count[i]++] = nr * SIZE + nc;
                }
            }
        }

        // 2. Precompute Bridge Responses
        int offsets[6][2] = {{-1,0}, {-1,1}, {0,-1}, {0,1}, {1,-1}, {1,0}};
        int pairs[3][2] = {{0, 5}, {1, 4}, {2, 3}}; // Opposite directions

        for (int i = 0; i < TOTAL_TILES; ++i) {
            int r = i / SIZE;
            int c = i % SIZE;
            
            for (auto& p : pairs) {
                int n1 = -1, n2 = -1;
                
                int r1 = r + offsets[p[0]][0], c1 = c + offsets[p[0]][1];
                if (r1>=0 && r1<SIZE && c1>=0 && c1<SIZE) n1 = r1 * SIZE + c1;

                int r2 = r + offsets[p[1]][0], c2 = c + offsets[p[1]][1];
                if (r2>=0 && r2<SIZE && c2>=0 && c2<SIZE) n2 = r2 * SIZE + c2;

                if (n1 != -1 && n2 != -1) {
                    bridge_partners[i][p[0]] = n2;
                    bridge_partners[i][p[1]] = n1;
                }
            }
        }
    }

    Move make_move(int turn, const std::vector<int>& board_vec, Move opp_move) {
        auto start_time = std::chrono::high_resolution_clock::now();
        pool_ptr = 0; // Reset Memory Pool

        // Convert board to raw array for speed
        int board[TOTAL_TILES];
        for(int i=0; i<TOTAL_TILES; ++i) board[i] = board_vec[i];

        // 0. OPENING
        if (turn == 1 && my_colour == RED) return Move(5, 5);
        if (turn == 2 && my_colour == BLUE && !is_swap(opp_move)) {
             if (opp_move.x >= 2 && opp_move.x <= 8 && opp_move.y >= 2 && opp_move.y <= 8) return Move(-1, -1);
        }

        // 1. TIME & LEGAL MOVES
        double time_limit = get_time_budget(board);
        std::vector<int> legal_indices;
        legal_indices.reserve(TOTAL_TILES);
        for(int i=0; i<TOTAL_TILES; ++i) if(board[i] == EMPTY) legal_indices.push_back(i);

        // 2. INSTANT WIN/LOSS CHECKS (Defensive)
        for (int idx : legal_indices) {
            board[idx] = my_colour;
            if (check_winner_full(board)) { // Check specific colour win
                 update_time(start_time);
                 return idx_to_move(idx);
            }
            board[idx] = EMPTY;
        }
        
        // Only run forced move checks if we are under pressure
        // (Optimized out for brevity, can be re-added if needed)

        // 3. MCTS ROOT
        int root_idx = pool_ptr++;
        node_pool[root_idx].init(-1, -1, legal_indices, opp_colour);

        int iterations = 0;
        int last_opp_idx = (is_swap(opp_move)) ? -1 : (opp_move.x * SIZE + opp_move.y);

        // 4. MCTS LOOP
        while (get_elapsed(start_time) < time_limit) {
            int node_idx = root_idx;
            
            // Local board copy for simulation
            // We use a fixed array stack to track changes for rollback? 
            // Actually, copying 121 ints is faster than complex rollback logic in simple MCTS
            int sim_board[TOTAL_TILES];
            std::memcpy(sim_board, board, TOTAL_TILES * sizeof(int));
            
            // Track moves for RAVE
            // Fixed size array is safer/faster than vector push_back
            int moves_in_tree[TOTAL_TILES];
            int tree_depth = 0;
            
            int current_player = my_colour;

            // --- SELECTION ---
            while (node_pool[node_idx].untried_moves.empty() && node_pool[node_idx].first_child != -1) {
                node_idx = best_child_rave(node_idx);
                int m = node_pool[node_idx].move_idx;
                moves_in_tree[tree_depth++] = m;
                sim_board[m] = node_pool[node_idx].player_just_moved;
                current_player = (current_player == RED) ? BLUE : RED;
            }

            // --- EXPANSION ---
            int last_sim_idx = last_opp_idx;
            if (!node_pool[node_idx].untried_moves.empty()) {
                if (pool_ptr < MAX_NODES - 5) { // Check pool safety
                    int rand_i = rng.next_int(node_pool[node_idx].untried_moves.size());
                    int move_idx = node_pool[node_idx].untried_moves[rand_i];
                    
                    // Fast remove
                    node_pool[node_idx].untried_moves[rand_i] = node_pool[node_idx].untried_moves.back();
                    node_pool[node_idx].untried_moves.pop_back();

                    sim_board[move_idx] = current_player;
                    
                    int child_idx = pool_ptr++;
                    node_pool[child_idx].init(move_idx, node_idx, node_pool[node_idx].untried_moves, current_player);
                    
                    // Link child
                    node_pool[child_idx].next_sibling = node_pool[node_idx].first_child;
                    node_pool[node_idx].first_child = child_idx;

                    node_idx = child_idx;
                    moves_in_tree[tree_depth++] = move_idx;
                    last_sim_idx = move_idx;
                    current_player = (current_player == RED) ? BLUE : RED;
                }
            }

            // --- SIMULATION (Optimized) ---
            int moves_in_sim[TOTAL_TILES];
            int sim_depth = 0;
            
            int winner = run_simulation_fast(sim_board, current_player, last_sim_idx, moves_in_sim, sim_depth);

            // --- BACKPROPAGATION ---
            while (node_idx != -1) {
                Node& n = node_pool[node_idx];
                n.visits++;
                if (n.player_just_moved == winner) n.wins++;

                // RAVE
                int child_idx = n.first_child;
                while (child_idx != -1) {
                    Node& c = node_pool[child_idx];
                    bool found = false;
                    // Check sim moves
                    for(int i=0; i<sim_depth; ++i) if(moves_in_sim[i] == c.move_idx) { found=true; break; }
                    // Check tree moves (if deeper)
                    if(!found) {
                        for(int i=0; i<tree_depth; ++i) if(moves_in_tree[i] == c.move_idx) { found=true; break; }
                    }
                    
                    if(found) {
                        c.amaf_visits++;
                        if (c.player_just_moved == winner) c.amaf_wins++;
                    }
                    child_idx = c.next_sibling;
                }
                node_idx = n.parent_idx;
            }
            iterations++;
        }

        // 5. SELECTION
        Move best_move(-1, -1);
        int max_visits = -1;
        double log_win_rate = 0.0;
        
        int child_idx = node_pool[root_idx].first_child;
        if (child_idx == -1) return idx_to_move(legal_indices[0]); // Fallback

        while (child_idx != -1) {
            if (node_pool[child_idx].visits > max_visits) {
                max_visits = node_pool[child_idx].visits;
                best_move = idx_to_move(node_pool[child_idx].move_idx);
                if (max_visits > 0) log_win_rate = node_pool[child_idx].wins / max_visits;
            }
            child_idx = node_pool[child_idx].next_sibling;
        }

        double time_taken = get_elapsed(start_time);
        update_time(start_time);

        std::cerr << "\n-------C++ Optimized: " 
                  << iterations << " iterations (" 
                  << std::fixed << std::setprecision(2) << time_taken << "s), WR: " 
                  << log_win_rate << "-------\n" << std::endl;

        return best_move;
    }

private:
    double get_elapsed(std::chrono::high_resolution_clock::time_point start) {
        std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - start;
        return diff.count();
    }
    void update_time(std::chrono::high_resolution_clock::time_point start) {
        total_time_used += get_elapsed(start);
    }
    double get_time_budget(int* board) {
        double remaining = (TOTAL_GAME_TIME - TIME_BUFFER) - total_time_used;
        int empty = 0;
        for(int i=0; i<TOTAL_TILES; ++i) if(board[i]==EMPTY) empty++;
        double turns = std::max(1.0, empty / 2.0);
        return std::max(std::min(remaining/turns, 5.0), 0.1);
    }

    int best_child_rave(int node_idx) {
        int best_child = -1;
        double best_score = -1e9;
        double log_visits = std::log(node_pool[node_idx].visits);

        int child_idx = node_pool[node_idx].first_child;
        while (child_idx != -1) {
            Node& c = node_pool[child_idx];
            double beta, uct, amaf;
            
            if (c.visits == 0) {
                beta = 1.0; uct = 0.5;
            } else {
                beta = RAVE_K / (RAVE_K + c.visits);
                uct = c.wins / c.visits;
            }
            
            amaf = (c.amaf_visits > 0) ? (c.amaf_wins / c.amaf_visits) : 0.5;
            double explore = (c.visits>0) ? EXPLORATION * std::sqrt(log_visits / c.visits) : 1.0;
            double score = (1.0 - beta) * uct + beta * amaf + explore;

            if (score > best_score) {
                best_score = score;
                best_child = child_idx;
            }
            child_idx = c.next_sibling;
        }
        return best_child;
    }

    // --- HOT LOOP: SIMULATION ---
    int run_simulation_fast(int* board, int turn, int last_idx, int* moves_made, int& depth) {
        // 1. Identify Empty Tiles efficiently
        int empty[TOTAL_TILES];
        int empty_count = 0;
        for(int i=0; i<TOTAL_TILES; ++i) if(board[i] == EMPTY) empty[empty_count++] = i;

        // 2. BRIDGE HEURISTIC (Optimized)
        if (last_idx != -1) {
            // Check neighbors of last opponent move
            int r = last_idx / SIZE;
            int c = last_idx % SIZE;
            int offsets[6][2] = {{-1,0}, {-1,1}, {0,-1}, {0,1}, {1,-1}, {1,0}};
            
            for(int k=0; k<6; ++k) {
                int nr = r + offsets[k][0];
                int nc = c + offsets[k][1];
                if (nr>=0 && nr<SIZE && nc>=0 && nc<SIZE) {
                    int n_idx = nr*SIZE + nc;
                    if (board[n_idx] == turn) { // My piece
                        int resp = bridge_partners[last_idx][k];
                        if (resp != -1 && board[resp] == EMPTY) {
                             // Must play here!
                             board[resp] = turn;
                             moves_made[depth++] = resp;
                             turn = (turn == RED) ? BLUE : RED;
                             
                             // Remove from empty list (Swap-remove)
                             for(int i=0; i<empty_count; ++i) {
                                 if(empty[i] == resp) {
                                     empty[i] = empty[--empty_count];
                                     break;
                                 }
                             }
                             goto random_fill;
                        }
                    }
                }
            }
        }
        
        random_fill:;
        
        // 3. RANDOM FILL (Fisher-Yates)
        while (empty_count > 0) {
            int rand_i = rng.next_int(empty_count);
            int idx = empty[rand_i];
            
            // Swap-remove
            empty[rand_i] = empty[--empty_count];
            
            board[idx] = turn;
            moves_made[depth++] = idx;
            turn = (turn == RED) ? BLUE : RED;
        }

        // 4. CHECK WINNER (Only Check RED)
        // If Red wins, return RED. Else return BLUE.
        return check_red_win_full(board) ? RED : BLUE;
    }

    bool check_red_win_full(const int* board) {
        int stack[TOTAL_TILES]; 
        int sp = 0;
        bool visited[TOTAL_TILES]; 
        std::memset(visited, 0, TOTAL_TILES); // Fast clear

        for (int start : red_starts) {
            if (board[start] == RED) {
                stack[sp++] = start;
                visited[start] = true;
            }
        }

        while (sp > 0) {
            int curr = stack[--sp];
            if (curr >= 110) return true; // Reached bottom

            int count = neighbor_count[curr];
            for (int k = 0; k < count; ++k) {
                int n = neighbors[curr][k];
                if (board[n] == RED && !visited[n]) {
                    visited[n] = true;
                    stack[sp++] = n;
                }
            }
        }
        return false;
    }

    // Helper: Standard check for non-full board (used in Step 2 of make_move)
    int check_winner_full(int* board) {
        if (check_red_win_full(board)) return RED;
        
        // Check Blue
        int stack[TOTAL_TILES]; int sp = 0;
        bool visited[TOTAL_TILES]; std::memset(visited, 0, TOTAL_TILES);
        for (int start : blue_starts) if (board[start] == BLUE) { stack[sp++] = start; visited[start] = true; }
        while (sp > 0) {
            int curr = stack[--sp];
            if (curr % SIZE == 10) return BLUE;
            int count = neighbor_count[curr];
            for(int k=0; k<count; ++k) {
                int n = neighbors[curr][k];
                if (board[n] == BLUE && !visited[n]) { visited[n]=true; stack[sp++] = n; }
            }
        }
        return 0;
    }

    bool is_swap(Move m) { return m.x == -1 && m.y == -1; }
    Move idx_to_move(int idx) { return Move(idx / SIZE, idx % SIZE); }
};

// --- PROTOCOL HANDLERS ---
std::vector<std::string> split(const std::string &s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) tokens.push_back(token);
    return tokens;
}

std::vector<int> parse_board_string(const std::string& board_str) {
    std::vector<int> board(TOTAL_TILES, 0);
    int idx = 0;
    for (char c : board_str) {
        if (c == 'R') board[idx++] = RED;
        else if (c == 'B') board[idx++] = BLUE;
        else if (c == '0') board[idx++] = EMPTY;
    }
    return board;
}

Move parse_move_string(const std::string& move_str) {
    if (move_str.empty()) return Move(-1, -1);
    std::vector<std::string> parts = split(move_str, ',');
    if (parts.size() != 2) return Move(-1, -1);
    return Move(std::stoi(parts[0]), std::stoi(parts[1]));
}

int main(int argc, char* argv[]) {
    if (argc < 2) return 1;
    BestAgent agent((argv[1][0] == 'R') ? RED : BLUE);
    std::string line;
    while (std::getline(std::cin, line)) {
        std::vector<std::string> parts = split(line, ';');
        if (parts.size() < 4) continue;
        Move opp_move = (parts[0] == "SWAP") ? Move(-1,-1) : parse_move_string(parts[1]);
        std::vector<int> board_vec = parse_board_string(parts[2]);
        Move best_move = agent.make_move(std::stoi(parts[3]), board_vec, opp_move);
        std::cout << best_move.x << "," << best_move.y << std::endl;
    }
    return 0;
}