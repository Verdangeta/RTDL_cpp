#include "rtdlite.h"
#include <iostream>
#include <cstdint>
#include <vector>
#include <climits>
#include <queue>
#include <map>
#include <string>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <iomanip>
#include <limits>
#include <cstdlib>
#include <cstring>

using namespace std;

// Internal type aliases (match rtdlite.h for ABI compatibility)
typedef rtd_value_t value_t;
typedef rtd_index_t index_t;
typedef rtd_birth_death_edges birth_death_edges;

/**
 * Disjoint Set Union (DSU) data structure with path compression and union by rank.
 * Used to efficiently track connected components in graphs during MST construction.
 */
class DSU {
  public:
	vector<index_t> parent;  // Parent array for each vertex
	vector<index_t> rank;   // Rank array for union by rank optimization

	/**
	 * Constructor: Initialize DSU with n_vertices disjoint sets.
	 * @param n_vertices Number of vertices in the graph
	 */
	DSU(index_t n_vertices) : parent(n_vertices), rank(n_vertices, 0) {
        iota(parent.begin(), parent.end(), 0);
	}

	/**
	 * Copy constructor: Create a copy of an existing DSU structure.
	 * @param copy_from DSU structure to copy from
	 */
    DSU(const DSU &copy_from) {
        parent.assign(copy_from.parent.begin(), copy_from.parent.end());
        rank.assign(copy_from.rank.begin(), copy_from.rank.end());
    }

	/**
	 * Find the root of the set containing vertex v with path compression.
	 * @param v Vertex to find root for
	 * @return Root vertex of the set containing v
	 */
	index_t find(index_t v) {
        if (parent[v] == v) {
            return v;
        }
        return parent[v] = find(parent[v]);
    }

	/**
	 * Unite two sets containing vertices u and v using union by rank.
	 * @param u First vertex
	 * @param v Second vertex
	 */
    void unite(index_t u, index_t v) {
        index_t u_root = find(u);
        index_t v_root = find(v);
        if (u_root != v_root) {
            if (rank[u_root] < rank[v_root]) {
                swap(u_root, v_root);
            }
            if (rank[u_root] == rank[v_root]) {
                ++rank[u_root];
            }
            parent[v_root] = u_root;
        }
    }
};

void prim_algo(const vector<vector<value_t>> &adjacency_matrix, vector<vector<int>> &mst_edges, vector<value_t> &edge_weights) {
    int n = (int)adjacency_matrix.size();

    vector<int> ancestors(n, -1);
    vector<value_t> dst(n, INT_MAX);
    vector<bool> visited(n, false);

    dst[0] = 0;
    ancestors[0] = -1;
    int v = 0;

    for (int iter = 0; iter < n; ++iter) {
        visited[v] = true;
        value_t mn_dst = INT_MAX;
        int next_vertex = -1;

        for (int i = 0; i < n; i++) {
            if (!visited[i] && adjacency_matrix[v][i] < dst[i]) {
                dst[i] = adjacency_matrix[v][i];
                ancestors[i] = v;
            }
            if (!visited[i] && dst[i] < mn_dst) {
                next_vertex = i;
                mn_dst = dst[i];
            }
        }

        // Failsafe for multiple connected components
        if (next_vertex == -1) {
            break;
        }
        v = next_vertex;
    }

    // Process all vertices, but skip unreachable ones (ancestors[i] == -1)
    // For disconnected graphs, some vertices may not be reachable
    int mst_pos = 0;
    for (int i = 1; i < n; i++) {
        if (ancestors[i] != -1) {  // Only process if vertex was reached
            mst_edges[mst_pos][0] = i;
            mst_edges[mst_pos][1] = ancestors[i];
            edge_weights[mst_pos] = adjacency_matrix[i][ancestors[i]];
            mst_pos++;
        } else {
            // For disconnected components, use a self-loop with max weight
            // This ensures the array has n-1 elements but marks unreachable vertices
            mst_edges[mst_pos][0] = i;
            mst_edges[mst_pos][1] = i;  // Self-loop
            edge_weights[mst_pos] = numeric_limits<value_t>::max();
            mst_pos++;
        }
    }
}

/**
 * RTD-Lite class for computing Relative Topological Distance between two distance matrices.
 * Computes topological barcodes that describe how connected components change when transitioning
 * from one distance metric (r1) to another (r2).
 */
class RTD_Lite {
    private:
        int n;                              // Number of vertices
        vector<vector<value_t>> r1;         // First distance matrix
        vector<vector<value_t>> r2;         // Second distance matrix
        vector<vector<value_t>> rmin;       // Element-wise minimum of r1 and r2

        /**
         * Compute element-wise minimum of r1 and r2 matrices, storing result in rmin.
         * Only computes upper triangular part and mirrors to lower triangular part.
         */
        void element_wise_min() {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j <= i; j++) {
                    rmin[j][i] = rmin[i][j] = min(r1[i][j], r2[i][j]);
                }
            }
        }

        /**
         * Sort edges by their weights in ascending order.
         * Both edge indices and edge weights are reordered accordingly.
         * 
         * @param edge_idx Vector of edge pairs (vertex indices) to be sorted
         * @param edge_w Vector of edge weights to be sorted
         */
        void sort_edges_weights(vector<vector<int>> &edge_idx, vector<value_t> &edge_w) {
            vector<int> idx(n - 1);
            iota(idx.begin(), idx.end(), 0);
            stable_sort(idx.begin(), idx.end(), [&edge_w](int i, int j) {return edge_w[i] < edge_w[j];});

            vector<vector<int>> tmp_edge_idx;
            tmp_edge_idx.assign(edge_idx.begin(), edge_idx.end());

            for (int i = 0; i < n - 1; i++) {
                edge_idx[i] = tmp_edge_idx[idx[i]];
            }

            vector<value_t> tmp_edge_w;
            tmp_edge_w.assign(edge_w.begin(), edge_w.end());
            for (int i = 0; i < n - 1; i++) {
                edge_w[i] = tmp_edge_w[idx[i]];
            }
        }

    public:

       // map<string, vector<vector<value_t>>> barcodes;
        map<string, vector<vector<int>>> barcodes_idx;  // Stores barcode intervals as vertex indices

        /**
         * Constructor: Initialize RTD-Lite with two distance matrices.
         * Computes element-wise minimum matrix rmin.
         * 
         * @param r1 First distance matrix (n x n)
         * @param r2 Second distance matrix (n x n)
         */
        RTD_Lite(const vector<vector<value_t>> &r1, const vector<vector<value_t>> &r2) : r1(r1), r2(r2) {
            n = (int)r1.size();
            rmin.resize(n, vector<value_t>(n));
            element_wise_min();
        }

        /**
         * Main computation method: Run RTD-Lite algorithm to compute topological barcodes.
         * 
         * Algorithm:
         * 1. Compute MSTs for rmin, r1, and r2
         * 2. Sort edges by weight for each MST
         * 3. For each edge in rmin MST, find when it connects components in r1 and r2
         * 4. Store birth-death intervals as vertex indices for valid barcodes
         */
        void run() {
            vector<vector<int>> rmin_edge_idx(n - 1, vector<int>(2));
            vector<value_t> rmin_edge_w(n - 1);
            prim_algo(rmin, rmin_edge_idx, rmin_edge_w);

            vector<vector<int>> r1_edge_idx(n - 1, vector<int>(2));
            vector<value_t> r1_edge_w(n - 1);
            prim_algo(r1, r1_edge_idx, r1_edge_w);
            sort_edges_weights(r1_edge_idx, r1_edge_w);
            
            run_with_r1_mst(rmin_edge_idx, rmin_edge_w, r1_edge_idx, r1_edge_w);
        }

        /**
         * Run RTD-Lite algorithm with precomputed r1 MST.
         * 
         * @param r1_edge_idx Precomputed r1 MST edges (n-1 edges, each is pair of vertex indices)
         * @param r1_edge_w Precomputed r1 MST edge weights (n-1 weights, already sorted)
         */
        void run(vector<vector<int>> &r1_edge_idx, vector<value_t> &r1_edge_w) {
            vector<vector<int>> rmin_edge_idx(n - 1, vector<int>(2));
            vector<value_t> rmin_edge_w(n - 1);
            prim_algo(rmin, rmin_edge_idx, rmin_edge_w);

            // Use provided r1 MST (assumed to be already sorted)
            run_with_r1_mst(rmin_edge_idx, rmin_edge_w, r1_edge_idx, r1_edge_w);
        }

    private:
        /**
         * Internal method: Run RTD-Lite algorithm with precomputed MSTs.
         * 
         * @param rmin_edge_idx rmin MST edges
         * @param rmin_edge_w rmin MST edge weights
         * @param r1_edge_idx r1 MST edges (already sorted)
         * @param r1_edge_w r1 MST edge weights (already sorted)
         */
        void run_with_r1_mst(vector<vector<int>> &rmin_edge_idx, vector<value_t> &rmin_edge_w,
                            vector<vector<int>> &r1_edge_idx, vector<value_t> &r1_edge_w) {
            vector<vector<int>> r2_edge_idx(n - 1, vector<int>(2));
            vector<value_t> r2_edge_w(n - 1);
            prim_algo(r2, r2_edge_idx, r2_edge_w);

            sort_edges_weights(rmin_edge_idx, rmin_edge_w);
            // r1_edge_idx and r1_edge_w are already sorted when passed
            sort_edges_weights(r2_edge_idx, r2_edge_w);

            DSU min_graph_dsu(n);

            barcodes_idx["1->2"] = vector<vector<int>>(n - 1, vector<int>(4));
            // Allocate space for n-1 regular barcodes + 1 potential max TSP edge
            barcodes_idx["2->1"] = vector<vector<int>>(n, vector<int>(4));

            for (int i = 0; i < n - 1; i++) {
                int u_clique = min_graph_dsu.find(rmin_edge_idx[i][0]);
                int v_clique = min_graph_dsu.find(rmin_edge_idx[i][1]);

                value_t birth = rmin_edge_w[i];
                // Initialize death values to birth (so condition death > birth will be false if not found)
                value_t death_1 = birth;
                value_t death_2 = birth;

                vector<int>* birth_idx = &rmin_edge_idx[i];
                vector<int>* death_1_idx = nullptr;  // Initialize to nullptr
                vector<int>* death_2_idx = nullptr;  // Initialize to nullptr

                DSU r1_graph_dsu(min_graph_dsu);
                for (int j = 0; j < n - 1; j++) {
                    r1_graph_dsu.unite(r1_edge_idx[j][0], r1_edge_idx[j][1]);
                    if (r1_graph_dsu.find(u_clique) == r1_graph_dsu.find(v_clique)) {
                        death_1 = r1_edge_w[j];
                        death_1_idx = &r1_edge_idx[j];
                        break;
                    }
                }

                DSU r2_graph_dsu(min_graph_dsu);
                for (int j = 0; j < n - 1; j++) {
                    r2_graph_dsu.unite(r2_edge_idx[j][0], r2_edge_idx[j][1]);
                    if (r2_graph_dsu.find(u_clique) == r2_graph_dsu.find(v_clique)) {
                        death_2 = r2_edge_w[j];
                        death_2_idx = &r2_edge_idx[j];
                        break;
                    }
                }

                // Only use death_1_idx if it was initialized (death_1 > birth implies it was found)
                if (death_1 > birth && death_1_idx != nullptr) {
         //           barcodes["1->2"][i] = vector<value_t>{birth, death_1};
                    barcodes_idx["1->2"][i] = vector<int>{birth_idx->at(0), birth_idx->at(1), death_1_idx->at(0), death_1_idx->at(1)};
                }
                // Only use death_2_idx if it was initialized (death_2 > birth implies it was found)
                if (death_2 > birth && death_2_idx != nullptr) {
           //         barcodes["2->1"][i] = vector<value_t>{birth, death_2};
                    barcodes_idx["2->1"][i] = vector<int>{birth_idx->at(0), birth_idx->at(1), death_2_idx->at(0), death_2_idx->at(1)};
                }
                min_graph_dsu.unite(rmin_edge_idx[i][0], rmin_edge_idx[i][1]);
            }

            // Find maximum edge in r2 (partial tour) - similar to max_TSP_row_col in Python
            // Mask out inf values (set to -inf) to find maximum among finite values
            value_t max_r2_value = -1.0;
            int max_r2_i = -1, max_r2_j = -1;
            bool found_max_r2 = false;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    if (i != j) {
                        value_t val = r2[i][j];
                        // Only consider finite values (not inf)
                        value_t inf_val = numeric_limits<value_t>::infinity();
                        if (val != inf_val && val == val && val > max_r2_value) {  // val == val checks for NaN
                            max_r2_value = val;
                            max_r2_i = i;
                            max_r2_j = j;
                            found_max_r2 = true;
                        }
                    }
                }
            }

            // Find biggest MST edge in r1 and birth_biggest_TSP_edge
            value_t biggest_MST_edge_w = 0.0;
            value_t birth_biggest_TSP_edge = 0.0;
            if (r1_edge_w.size() > 0) {
                // Find maximum edge weight in r1 MST
                biggest_MST_edge_w = *max_element(r1_edge_w.begin(), r1_edge_w.end());
                
                // Find minimum edge in r1 that is larger than biggest_MST_edge_w
                value_t min_valid_edge = numeric_limits<value_t>::max();
                bool found_valid_edge = false;
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        value_t r1_val = r1[i][j];
                        value_t inf_val = numeric_limits<value_t>::infinity();
                        if (i != j && r1_val != inf_val && r1_val == r1_val && r1_val > biggest_MST_edge_w && r1_val < min_valid_edge) {
                            min_valid_edge = r1[i][j];
                            found_valid_edge = true;
                        }
                    }
                }
                if (found_valid_edge) {
                    birth_biggest_TSP_edge = min_valid_edge;
                } else {
                    birth_biggest_TSP_edge = biggest_MST_edge_w;
                }
            }

            // Add max TSP edge barcode if valid
            if (found_max_r2) {
                value_t max_edge_weight = max(max_r2_value - birth_biggest_TSP_edge, 0.0);
                const value_t eps = 1e-7;
                if (max_edge_weight > eps) {
                    // Find an edge in r1 with weight close to birth_biggest_TSP_edge for birth edge
                    // This edge represents the "birth" threshold
                    int birth_edge_i = -1, birth_edge_j = -1;
                    value_t min_diff = numeric_limits<value_t>::max();
                    for (int i = 0; i < n; i++) {
                        for (int j = 0; j < n; j++) {
                            value_t r1_val = r1[i][j];
                            value_t inf_val = numeric_limits<value_t>::infinity();
                            if (i != j && r1_val != inf_val && r1_val == r1_val) {
                                value_t diff = abs(r1_val - birth_biggest_TSP_edge);
                                if (diff < min_diff && r1_val >= birth_biggest_TSP_edge) {
                                    min_diff = diff;
                                    birth_edge_i = i;
                                    birth_edge_j = j;
                                }
                            }
                        }
                    }
                    // If no exact match found, use the first valid edge as fallback
                    if (birth_edge_i == -1) {
                        birth_edge_i = max_r2_i;
                        birth_edge_j = max_r2_j;
                    }
                    barcodes_idx["2->1"][n - 1] = vector<int>{birth_edge_i, birth_edge_j, max_r2_i, max_r2_j};
                }
            }
        }
};

const value_t EPS = 1e-7;  // Epsilon value for floating-point comparisons

/**
 * Internal: run RTD-Lite computation and format results.
 */
static rtd_lite_result run_main(vector<vector<value_t>> &r1, vector<vector<value_t>> &r2, int n, bool return_indices,
                         vector<vector<int>> *r1_mst_edge_idx = nullptr, vector<value_t> *r1_mst_edge_w = nullptr) {
    RTD_Lite rtd_lite_instance(r1, r2);
    if (r1_mst_edge_idx != nullptr && r1_mst_edge_w != nullptr) {
        rtd_lite_instance.run(*r1_mst_edge_idx, *r1_mst_edge_w);
    } else {
        rtd_lite_instance.run();
    }

    int intervals_left = 0;
    for (int i = 0; i < (int)rtd_lite_instance.barcodes_idx["1->2"].size(); ++i) {
        if (rtd_lite_instance.barcodes_idx["1->2"][i][0] + rtd_lite_instance.barcodes_idx["1->2"][i][1] > 0) {
            ++intervals_left;
        }
    }

    int intervals_right = 0;
    for (int i = 0; i < (int)rtd_lite_instance.barcodes_idx["2->1"].size(); ++i) {
        if (rtd_lite_instance.barcodes_idx["2->1"][i][0] + rtd_lite_instance.barcodes_idx["2->1"][i][1] > 0) {
            ++intervals_right;
        }
    }

    rtd_birth_death_edges* left_to_right = (intervals_left > 0)
        ? (rtd_birth_death_edges*)malloc(sizeof(rtd_birth_death_edges) * intervals_left) : nullptr;
    rtd_birth_death_edges* right_to_left = (intervals_right > 0)
        ? (rtd_birth_death_edges*)malloc(sizeof(rtd_birth_death_edges) * intervals_right) : nullptr;

    int pos = 0;
    for (int i = 0; i < (int)rtd_lite_instance.barcodes_idx["1->2"].size(); ++i) {
        if (rtd_lite_instance.barcodes_idx["1->2"][i][0] + rtd_lite_instance.barcodes_idx["1->2"][i][1] > 0) {
            left_to_right[pos++] = {(rtd_index_t)rtd_lite_instance.barcodes_idx["1->2"][i][0],
                (rtd_index_t)rtd_lite_instance.barcodes_idx["1->2"][i][1],
                (rtd_index_t)rtd_lite_instance.barcodes_idx["1->2"][i][2],
                (rtd_index_t)rtd_lite_instance.barcodes_idx["1->2"][i][3]};
        }
    }

    pos = 0;
    for (int i = 0; i < (int)rtd_lite_instance.barcodes_idx["2->1"].size(); ++i) {
        if (rtd_lite_instance.barcodes_idx["2->1"][i][0] + rtd_lite_instance.barcodes_idx["2->1"][i][1] > 0) {
            right_to_left[pos++] = {(rtd_index_t)rtd_lite_instance.barcodes_idx["2->1"][i][0],
                (rtd_index_t)rtd_lite_instance.barcodes_idx["2->1"][i][1],
                (rtd_index_t)rtd_lite_instance.barcodes_idx["2->1"][i][2],
                (rtd_index_t)rtd_lite_instance.barcodes_idx["2->1"][i][3]};
        }
    }

    return {intervals_left, intervals_right, left_to_right, right_to_left};
}

/* --- Public C API (see rtdlite.h) --- */

extern "C" void rtd_lite_result_free(rtd_lite_result *result) {
    if (result == nullptr) return;
    free(result->left_to_right);
    free(result->right_to_left);
    result->left_to_right = nullptr;
    result->right_to_left = nullptr;
    result->left_bars = 0;
    result->right_bars = 0;
}

extern "C" rtd_lite_result rtd_lite_run_matrix(rtd_value_t *data_1, rtd_value_t *data_2,
                                                int num_vertices, bool return_indices) {
    int n = num_vertices;
    vector<vector<value_t>> r1(n, vector<value_t>(n));
    vector<vector<value_t>> r2(n, vector<value_t>(n));

    for (int i = 0; i < n; ++i) {
        r1[i].assign(data_1 + i * n, data_1 + (i + 1) * n);
        r2[i].assign(data_2 + i * n, data_2 + (i + 1) * n);
    }

    return run_main(r1, r2, n, return_indices);
}

extern "C" rtd_lite_result rtd_lite_run_matrix_with_mst(rtd_value_t *data_1, rtd_value_t *data_2,
                                                        int num_vertices,
                                                        int *r1_mst_edge_idx, rtd_value_t *r1_mst_edge_w,
                                                        bool return_indices) {
    int n = num_vertices;
    vector<vector<value_t>> r1(n, vector<value_t>(n));
    vector<vector<value_t>> r2(n, vector<value_t>(n));

    for (int i = 0; i < n; ++i) {
        r1[i].assign(data_1 + i * n, data_1 + (i + 1) * n);
        r2[i].assign(data_2 + i * n, data_2 + (i + 1) * n);
    }

    vector<vector<int>> r1_mst_idx(n - 1, vector<int>(2));
    vector<value_t> r1_mst_w(n - 1);
    for (int i = 0; i < n - 1; ++i) {
        r1_mst_idx[i][0] = r1_mst_edge_idx[i * 2];
        r1_mst_idx[i][1] = r1_mst_edge_idx[i * 2 + 1];
        r1_mst_w[i] = r1_mst_edge_w[i];
    }

    return run_main(r1, r2, n, return_indices, &r1_mst_idx, &r1_mst_w);
}

extern "C" rtd_lite_result rtd_lite_run_from_file(const char *filename, int num_vertices, bool return_indices) {
    int n = num_vertices;
    vector<vector<value_t>> r1(n, vector<value_t>(n));
    vector<vector<value_t>> r2(n, vector<value_t>(n));

    ifstream file(filename);
    if (file.is_open()) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                file >> r1[j][k];
            }
        }
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                file >> r2[j][k];
            }
        }
        file.close();
    }

    return run_main(r1, r2, n, return_indices);
}

/* --- Legacy names for Python ctypes compatibility --- */
extern "C" rtd_lite_result run_matrix(value_t *data_1, value_t *data_2, int num_vertices, bool return_indices) {
    return rtd_lite_run_matrix(data_1, data_2, num_vertices, return_indices);
}

extern "C" rtd_lite_result run_matrix_with_mst(value_t *data_1, value_t *data_2, int num_vertices,
                                               int *r1_mst_edge_idx, value_t *r1_mst_edge_w, bool return_indices) {
    return rtd_lite_run_matrix_with_mst(data_1, data_2, num_vertices,
                                        r1_mst_edge_idx, r1_mst_edge_w, return_indices);
}


#ifndef RTDLITE_LIBRARY
/**
 * Main entry point for command-line execution (excluded when built as library).
 * Usage: ./rtdlite <matrix_filename> <num_vertices>
 */
int main(int argc, char* argv[]) {
    if (argc < 3) {
        cerr << "Usage: " << (argc > 0 ? argv[0] : "rtdlite") << " <matrix_filename> <num_vertices>" << endl;
        return 1;
    }

    const char* matrix_filename = argv[1];
    int n = atoi(argv[2]);

    rtd_lite_result result = rtd_lite_run_from_file(matrix_filename, n, true);

    ofstream out_file("out.txt");
    if (out_file.is_open()) {
        for (int64_t j = 0; j < result.left_bars; j++) {
            out_file << fixed << setprecision(16)
                << result.left_to_right[j].birth_i << " " << result.left_to_right[j].birth_j
                << " " << result.left_to_right[j].death_i << " " << result.left_to_right[j].death_j << "\n";
        }
        for (int64_t j = 0; j < result.right_bars; j++) {
            out_file << fixed << setprecision(16)
                << result.right_to_left[j].birth_i << " " << result.right_to_left[j].birth_j
                << " " << result.right_to_left[j].death_i << " " << result.right_to_left[j].death_j << "\n";
        }
        out_file.close();
    }

    rtd_lite_result_free(&result);
    return 0;
}
#endif /* RTDLITE_LIBRARY */
