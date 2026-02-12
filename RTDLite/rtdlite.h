/**
 * @file rtdlite.h
 * @brief RTD-Lite: Relative Topological Distance computation for two distance matrices
 *
 * Computes topological barcodes that describe correspondence between connected
 * components when transitioning from one distance metric (r1) to another (r2).
 * Key use case: r1 = full graph, r2 = partial tour (TSP) with inf for missing edges.
 *
 * @author RTDLite contributors
 * @license MIT
 */

#ifndef RTDLITE_H
#define RTDLITE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

/** Floating point type for distance values */
typedef double rtd_value_t;

/** Integer type for vertex/edge indices */
typedef int64_t rtd_index_t;

/**
 * Birth-death edges: vertex indices of the birth and death edges of a barcode interval.
 * birth_i, birth_j = endpoints of the birth edge (from rmin/r1 MST)
 * death_i, death_j = endpoints of the death edge (from r1 or r2 MST)
 */
typedef struct {
    rtd_index_t birth_j;
    rtd_index_t birth_i;
    rtd_index_t death_i;
    rtd_index_t death_j;
} rtd_birth_death_edges;

/**
 * Result of RTD-Lite computation.
 * left_to_right and right_to_left are heap-allocated; caller must free with rtd_lite_result_free().
 */
typedef struct {
    rtd_index_t left_bars;              /**< Number of barcodes for direction 1->2 */
    rtd_index_t right_bars;             /**< Number of barcodes for direction 2->1 */
    rtd_birth_death_edges *left_to_right;  /**< Barcodes 1->2: (birth from r1, death from r1) */
    rtd_birth_death_edges *right_to_left;  /**< Barcodes 2->1: (birth from r1, death from r2) */
} rtd_lite_result;

/**
 * Free memory allocated by rtd_lite_run_* functions.
 * @param result Result structure whose arrays were allocated by the library
 */
void rtd_lite_result_free(rtd_lite_result *result);

/**
 * Run RTD-Lite from two distance matrices (flat arrays, row-major, n*n elements).
 *
 * @param data_1 First distance matrix (r1), e.g. full graph
 * @param data_2 Second distance matrix (r2), e.g. partial tour (inf for missing edges)
 * @param num_vertices Matrix dimension n
 * @param return_indices If true, result contains vertex indices (always true in practice)
 * @return Result with allocated arrays; caller must call rtd_lite_result_free()
 */
rtd_lite_result rtd_lite_run_matrix(rtd_value_t *data_1, rtd_value_t *data_2,
                                    int num_vertices, bool return_indices);

/**
 * Run RTD-Lite with precomputed r1 MST (avoids recomputing MST when r1 is fixed).
 *
 * @param data_1 First distance matrix
 * @param data_2 Second distance matrix
 * @param num_vertices Matrix dimension n
 * @param r1_mst_edge_idx Flat array [u0,v0, u1,v1, ...] of n-1 edges (vertex indices)
 * @param r1_mst_edge_w Flat array of n-1 edge weights, already sorted ascending
 * @param return_indices Whether to return vertex indices
 * @return Result; caller must call rtd_lite_result_free()
 */
rtd_lite_result rtd_lite_run_matrix_with_mst(rtd_value_t *data_1, rtd_value_t *data_2,
                                             int num_vertices,
                                             int *r1_mst_edge_idx, rtd_value_t *r1_mst_edge_w,
                                             bool return_indices);

/**
 * Run RTD-Lite from file. Format: n*n values for r1, then n*n values for r2 (whitespace-separated).
 *
 * @param filename Path to file
 * @param num_vertices Matrix dimension n
 * @param return_indices Whether to return vertex indices
 * @return Result; caller must call rtd_lite_result_free()
 */
rtd_lite_result rtd_lite_run_from_file(const char *filename, int num_vertices, bool return_indices);

#ifdef __cplusplus
}
#endif

#endif /* RTDLITE_H */
