from __future__ import print_function
import ctypes
import sys
from .converter import convert, convert_with_mst, printHelpAndExit
from .converter import find
import numpy as np
import scipy.sparse as sps
import os
import platform
import pathlib
import ctypes.util
import torch
import torch.nn as nn

from copy import deepcopy
from sklearn.metrics import pairwise_distances


def _native_lib_candidates():
    """
    Return candidate native library filenames and glob patterns by OS.
    Supports both historical and current naming conventions.
    """
    system = platform.system()
    if system == "Windows":
        # Keep compatibility with direct DLL loading.
        return ["rtd_lite.dll", "librtd_lite.dll", "rtd_lite*.pyd", "rtd_lite*.dll"]
    if system == "Darwin":
        return ["rtd_lite.dylib", "librtd_lite.dylib", "rtd_lite*.dylib", "librtd_lite*.dylib"]
    # Linux and other Unix-like systems.
    return ["rtd_lite.so", "librtd_lite.so", "rtd_lite*.so", "librtd_lite*.so"]


def _load_rtd_lite_library():
    """
    Load RTD-Lite native library from package directory with robust fallback.
    """
    module_dir = pathlib.Path(__file__).resolve().parent
    candidates = []
    seen = set()

    for pattern in _native_lib_candidates():
        # If pattern has wildcard, expand it; otherwise treat as direct file.
        if any(ch in pattern for ch in "*?[]"):
            expanded = sorted(module_dir.glob(pattern))
            for path in expanded:
                key = str(path)
                if key not in seen:
                    seen.add(key)
                    candidates.append(path)
        else:
            path = module_dir / pattern
            key = str(path)
            if key not in seen:
                seen.add(key)
                candidates.append(path)

    load_errors = []
    for path in candidates:
        if not path.exists():
            continue
        try:
            return ctypes.cdll.LoadLibrary(str(path))
        except OSError as exc:
            load_errors.append(f"{path}: {exc}")

    # Optional system-level fallback by soname.
    for libname in ("rtd_lite", "librtd_lite"):
        found = ctypes.util.find_library(libname)
        if not found:
            continue
        try:
            return ctypes.cdll.LoadLibrary(found)
        except OSError as exc:
            load_errors.append(f"{found}: {exc}")

    searched = ", ".join(str(p) for p in candidates)
    errors = "\n".join(load_errors) if load_errors else "No compatible library file found."
    raise RuntimeError(
        "Could not load RTD-Lite native library.\n"
        f"Searched in: {searched}\n"
        f"Load errors:\n{errors}"
    )


def run(cloud_1 = None, cloud_2 = None): 
    matrix_1, matrix_2 = None, None
    file_name = ""
    if cloud_1 is not None and isinstance(cloud_1, str):
        file_name = ctypes.c_char_p(cloud_1.encode('utf-8'))
    elif cloud_1 is not None and isinstance(cloud_1, np.ndarray):
        matrix_1 = cloud_1
        matrix_2 = cloud_2
    elif cloud_1 is not None and isinstance(cloud_1, sps.coo_matrix):
        matrix_1 = cloud_1
        matrix_2 = cloud_2
    else:
        printHelpAndExit("Error: First argument must either be a string for file name, or a numpy array for input data")
    
    #if matrix_1:
    prog = None

    prog = _load_rtd_lite_library()

    rank = convert(prog, file_name, matrix_1, matrix_2)
    return rank
    
    

## Prim's algorithm only for total weight (without returning actual edges)

def prim_algo_simplified(adjacency_matrix):
    n = len(adjacency_matrix)
    
    infty = torch.max(adjacency_matrix).item() + 10
    dst = torch.ones(n, device=adjacency_matrix.device) * infty
    ancestors = -torch.ones(n, dtype=int, device=adjacency_matrix.device)
    visited = torch.zeros(n, dtype=bool, device=adjacency_matrix.device)
    
    s, v = torch.tensor(0.0, device=adjacency_matrix.device), 0
    for i in range(n - 1):
        visited[v] = 1
        
        ancestors[dst > adjacency_matrix[v]] = v
        dst = torch.minimum(dst, adjacency_matrix[v])
        dst[visited] = infty
        v = torch.argmin(dst)
       
        s += adjacency_matrix[v, ancestors[v]]

    return s


### Main part
class RTD_Lite:
    def __init__(self, r1, r2, quant_outer=None, quant_inner=None, distance='euclidean'):
        self.prog = None
        self.prog = _load_rtd_lite_library()
        
        if distance == 'euclidean':
            dists_1 = torch.cdist(r1, r1)
            dists_2 = torch.cdist(r2, r2)
        elif distance == 'precomputed':
            dists_1 = r1
            dists_2 = r2
        else:
            printHelpAndExit("Only distance='euclidean' or distance='precomputed' are supported.")
        
        
        if quant_outer is None:
            quant_outer = torch.quantile(dists_1, 0.9)
        self.r1 = dists_1 / quant_outer
        
        
        if quant_inner is None:
            quant_inner = torch.quantile(dists_2, 0.9)
        self.r2 = dists_2 / quant_inner
        self.device = r1.device

        
    def __call__(self, r1_mst=None):
        """
        Run RTD-Lite computation.
        
        @param r1_mst Optional: tuple (r1_edge_idx, r1_edge_w) where:
            - r1_edge_idx: numpy array (n-1, 2) with MST edge indices (already sorted by weight)
            - r1_edge_w: numpy array (n-1,) with MST edge weights (already sorted)
        @return Dictionary with barcodes {'1->2': tensor, '2->1': tensor}
        """
        if r1_mst is not None:
            r1_edge_idx, r1_edge_w = r1_mst
            # Convert to numpy if needed
            if isinstance(r1_edge_idx, torch.Tensor):
                r1_edge_idx = r1_edge_idx.cpu().numpy()
            if isinstance(r1_edge_w, torch.Tensor):
                r1_edge_w = r1_edge_w.cpu().numpy()
            barcode_l, barcode_r = convert_with_mst(self.prog, self.r1.cpu().numpy(), self.r2.cpu().numpy(),
                                                     r1_edge_idx, r1_edge_w)
        else:
            barcode_l, barcode_r = convert(self.prog, "", self.r1.cpu().numpy(), self.r2.cpu().numpy())
     
        # Store raw barcode_r for later use (to avoid recomputing convert_with_mst)
        self._last_barcode_r_np = np.asarray(barcode_r, dtype=np.int64)
        if self._last_barcode_r_np.ndim == 1 and self._last_barcode_r_np.shape[0] == 4:
            self._last_barcode_r_np = self._last_barcode_r_np.reshape(1, 4)
     
        # We need it for the reconstruction
        rmin = torch.minimum(self.r1, self.r2)
        
        barcodes = {'1->2' : [], '2->1' : []}
        barcode_l = torch.from_numpy(barcode_l).long()
        barcode_r = torch.from_numpy(barcode_r).long()
        
        # Handle empty barcodes
        if barcode_l.shape[0] > 0:
            barcodes['1->2'] = torch.stack([rmin[barcode_l[:, 0], barcode_l[:, 1]], 
                                         self.r1[barcode_l[:, 2], barcode_l[:, 3]] ], -1)
        else:
            barcodes['1->2'] = torch.tensor([], dtype=torch.float32, device=self.device).reshape(0, 2)
        
        if barcode_r.shape[0] > 0:
            # Extract birth and death values
            birth_vals = rmin[barcode_r[:, 0], barcode_r[:, 1]]
            death_vals = self.r2[barcode_r[:, 2], barcode_r[:, 3]]
            
            # Fix inf values in death: if death is inf, find max finite value in r2
            # This handles the case where max TSP edge indices point to inf in r2
            inf_mask = torch.isinf(death_vals)
            if torch.any(inf_mask):
                # Find maximum finite value in r2 (mask out inf)
                masked_r2 = torch.where(torch.isinf(self.r2), torch.tensor(float('-inf'), device=self.device), self.r2)
                if torch.any(~torch.isinf(masked_r2)):
                    max_r2_val = torch.max(masked_r2)
                    # Replace inf death values with the max finite value from r2
                    death_vals[inf_mask] = max_r2_val
            
            barcodes['2->1'] = torch.stack([birth_vals, death_vals], -1)
        else:
            barcodes['2->1'] = torch.tensor([], dtype=torch.float32, device=self.device).reshape(0, 2)

        # Store last computed barcodes for get_edge_weights
        self._last_barcodes_21 = barcodes['2->1']

        return barcodes
    
    def get_edge_weights(self, tour_edges=None, barcodes_21=None):
        """
        Get edge weights as tensors directly from computed barcodes.
        Optimized version that returns tensors instead of dictionary.
        
        @param tour_edges: Optional list of tuples [(u, v), ...] representing edges in partial tour.
                          If None, returns all edges from barcodes.
        @param barcodes_21: Optional precomputed barcodes['2->1'] tensor. If None, uses last computed from __call__.
        @return: Tuple (edge_indices, edge_weights) where:
                 - edge_indices: torch.Tensor [N, 2] with edge indices (u, v)
                 - edge_weights: torch.Tensor [N] with RTDL weights
                 If tour_edges is None, returns all edges. Otherwise filters by tour_edges.
        """
        if barcodes_21 is None:
            # Use last computed barcodes if available
            if not hasattr(self, '_last_barcodes_21'):
                raise ValueError("barcodes_21 must be provided or __call__ must be called first")
            barcodes_21 = self._last_barcodes_21
        
        barcode_r = self._last_barcode_r_np
        if barcode_r.size == 0:
            if tour_edges:
                edge_indices = torch.tensor(tour_edges, dtype=torch.long, device=self.device)
                edge_weights = torch.zeros(len(tour_edges), dtype=torch.float32, device=self.device)
                return edge_indices, edge_weights
            else:
                return torch.empty((0, 2), dtype=torch.long, device=self.device), torch.empty(0, dtype=torch.float32, device=self.device)
        
        # Use torch tensors directly
        device = self.device
        if isinstance(barcodes_21, torch.Tensor):
            barcodes_21_tensor = barcodes_21.to(device)
        else:
            barcodes_21_tensor = torch.tensor(barcodes_21, device=device, dtype=torch.float32)
        
        if barcodes_21_tensor.numel() == 0:
            if tour_edges:
                edge_indices = torch.tensor(tour_edges, dtype=torch.long, device=self.device)
                edge_weights = torch.zeros(len(tour_edges), dtype=torch.float32, device=self.device)
                return edge_indices, edge_weights
            else:
                return torch.empty((0, 2), dtype=torch.long, device=self.device), torch.empty(0, dtype=torch.float32, device=self.device)
        
        # Ensure 2D shape
        if barcodes_21_tensor.dim() == 1:
            if barcodes_21_tensor.shape[0] == 2:
                barcodes_21_tensor = barcodes_21_tensor.unsqueeze(0)
            elif barcodes_21_tensor.shape[0] % 2 == 0:
                barcodes_21_tensor = barcodes_21_tensor.reshape(-1, 2)
            else:
                barcodes_21_tensor = torch.tensor([], dtype=torch.float32, device=device).reshape(0, 2)
        
        # death edge indices: (death_i, death_j) - already in barcode_r[:, 2:4]
        path_edges_from_barcodes = torch.from_numpy(barcode_r[:, 2:4].astype(np.int64)).to(device)
        
        n_vertices = self.r1.shape[0]
        total_barcode_count = min(len(path_edges_from_barcodes), barcodes_21_tensor.shape[0])
        
        if total_barcode_count == 0:
            if tour_edges:
                edge_indices = torch.tensor(tour_edges, dtype=torch.long, device=device)
                edge_weights = torch.zeros(len(tour_edges), dtype=torch.float32, device=device)
                return edge_indices, edge_weights
            else:
                return torch.empty((0, 2), dtype=torch.long, device=device), torch.empty(0, dtype=torch.float32, device=device)
        
        # Compute weights directly from tensors: death - birth
        # Process all barcodes together (regular n-1 + optional max TSP edge)
        weights = barcodes_21_tensor[:total_barcode_count, 1] - barcodes_21_tensor[:total_barcode_count, 0]
        edge_indices_all_barcodes = path_edges_from_barcodes[:total_barcode_count]
        
        # Filter non-zero weights
        nonzero_mask = torch.abs(weights) > 1e-9
        edge_indices_filtered = edge_indices_all_barcodes[nonzero_mask]
        weights_filtered = weights[nonzero_mask]
        
        # Add reverse edges for all barcodes
        if len(edge_indices_filtered) > 0:
            edge_indices_reversed = edge_indices_filtered.flip(dims=[1])  # Swap u and v
            edge_indices_all = torch.cat([edge_indices_filtered, edge_indices_reversed], dim=0)
            weights_all = torch.cat([weights_filtered, weights_filtered], dim=0)
        else:
            edge_indices_all = torch.empty((0, 2), dtype=torch.long, device=device)
            weights_all = torch.empty(0, dtype=torch.float32, device=device)
        
        # Filter by tour_edges if provided
        if tour_edges is not None:
            tour_edges_tensor = torch.tensor(tour_edges, dtype=torch.long, device=device)
            
            # Handle empty edge_indices_all case
            if edge_indices_all.shape[0] == 0:
                # No edges available, return zeros for all tour edges
                tour_weights = torch.zeros(len(tour_edges), dtype=torch.float32, device=device)
                return tour_edges_tensor, tour_weights
            
            # Use vectorized lookup: for each tour edge, find matching index in edge_indices_all
            # Expand dimensions for broadcasting: tour_edges [N_tour, 1, 2] vs edge_indices_all [1, N_cache, 2]
            tour_edges_expanded = tour_edges_tensor.unsqueeze(1)  # [len(tour_edges), 1, 2]
            edge_indices_expanded = edge_indices_all.unsqueeze(0)  # [1, N, 2]
            
            # Find matches: (tour_edges_expanded == edge_indices_expanded) gives [len(tour_edges), N, 2]
            # Match when both coordinates are equal
            matches = (tour_edges_expanded == edge_indices_expanded).all(dim=2)  # [len(tour_edges), N]
            
            # For each tour edge, find first matching index (or 0 if no match, but we'll mask it)
            match_indices = matches.long().argmax(dim=1)  # [len(tour_edges)] - index of first match
            has_match = matches.any(dim=1)  # [len(tour_edges)] - whether match exists
            
            # Extract weights: use matched indices, or 0.0 if no match
            tour_weights = torch.where(
                has_match,
                weights_all[match_indices],
                torch.zeros(len(tour_edges), dtype=torch.float32, device=device)
            )
            
            return tour_edges_tensor, tour_weights
        else:
            return edge_indices_all, weights_all
    
class RTD_Lite_summ_only:
    def __init__(self, r1, r2, quant_outer=None, quant_inner=None, distance='euclidean'):
        dists_1 = torch.cdist(r1, r1)
        if quant_outer is None:
            quant_outer = torch.quantile(dists_1, 0.9)
        self.r1 = dists_1 / quant_outer
        
        dists_2 = torch.cdist(r2, r2)
        if quant_inner is None:
            quant_inner = torch.quantile(dists_2, 0.9)
        self.r2 = dists_2 / quant_inner
        
        self.device = r1.device
        
    def __call__(self):
        rmin = torch.minimum(self.r1, self.r2)
        
        rmin_sum = prim_algo_simplified(rmin.cpu())
        r1_sum = prim_algo_simplified(self.r1.cpu())
        r2_sum = prim_algo_simplified(self.r2.cpu())

        return 0.5 * (r1_sum - rmin_sum + r2_sum - rmin_sum) 