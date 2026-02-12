from __future__ import print_function
import numpy as np
import scipy.sparse as sps
import ctypes
import math
import sys
import re
import os

'''
Prints out the error message and quits the program.
msg -- Custom error message to show the user
'''
def printHelpAndExit(msg):
    error_msg = msg + '''
    How to use this:
    parallelrank package:
    User Functions:
        run(matrix or file_name, matrix or file_name, dist='precomputed')
                First Argument: Could be either of the following but not both
                    matrix: Must be a 2-dimensional numpy array
                    file_name: Must be of type string
                Second Argument: Could be either of the following but not both
                    matrix: Must be a 2-dimensional numpy array
                    file_name: Must be of type string
                    
    For more information, please see README.md.
    '''

    raise Exception(error_msg)
'''
Searches the path and all its children for file named name
'''
def find(name, path):   #stackoverflow.com/questions/1724693/find-a-file-in-python
    for root,dirs,files in os.walk(path):
        if name in files:
            return os.path.join(root,name)

class Birth_death_edges(ctypes.Structure):
    """
    Replica of datatype for bacrode from cuda
    """
    pass
    _fields_ = [("birth_i",ctypes.c_int64),("birth_j",ctypes.c_int64),("death_i",ctypes.c_int64),("death_j",ctypes.c_int64)]

class RTD_Lite_result(ctypes.Structure):
    """
    Replica of datatype for result from cuda
    """
    pass
    _fields_ = [("left_bars",ctypes.c_int64), ("right_bars",ctypes.c_int64), ("left_to_right",ctypes.POINTER(Birth_death_edges)), ("right_to_left",ctypes.POINTER(Birth_death_edges))]

def _bd_edges_to_array(bd_pair):
    return np.array([bd_pair.birth_i, bd_pair.birth_j, bd_pair.death_i, bd_pair.death_j])

def convert(prog, file_name, user_matrix = None, user_matrix_2 = None, num_vertices=-1):

    if isinstance(user_matrix, np.ndarray) and len(user_matrix) == 0:
        user_matrix = None
    elif isinstance(user_matrix, sps.coo_matrix) and sps.coo_matrix.getnnz(user_matrix) == 0:
        user_matrix = None
 
    # Read from file if not given
    if user_matrix is None:           
        prog.run_main_filename.restype = RTD_Lite_result
        res = prog.run_main_filename(n_threads, file_name, (ctypes.c_int64)(num_vertices), (ctypes.c_bool)(True))
    else:
        num_rows, num_columns, num_entries, user_matrix = distance_matrix_user_matrix(user_matrix)
        if user_matrix is None:
            printHelpAndExit("Matrix was not created")
            return

        user_matrix = (ctypes.c_double * num_entries)(*user_matrix)
        
        user_matrix_2 = (ctypes.c_double * num_entries)(*user_matrix_2.ravel())
        
        prog.run_matrix.restype = RTD_Lite_result
        res = prog.run_matrix(user_matrix, user_matrix_2, (ctypes.c_int64)(num_rows), (ctypes.c_bool)(True))

     
    # Ensure 2D arrays with shape (n, 4) even for empty or single barcode cases
    if res.left_bars == 0:
        barcode_left = np.empty((0, 4), dtype=np.int64)
    else:
        barcode_left = np.array([_bd_edges_to_array(res.left_to_right[NUM]) for NUM in range(res.left_bars)])
        # Ensure 2D shape: if single barcode, reshape from (4,) to (1, 4)
        if barcode_left.ndim == 1:
            barcode_left = barcode_left.reshape(1, 4)
    
    if res.right_bars == 0:
        barcode_right = np.empty((0, 4), dtype=np.int64)
    else:
        barcode_right = np.array([_bd_edges_to_array(res.right_to_left[NUM]) for NUM in range(res.right_bars)])
        # Ensure 2D shape: if single barcode, reshape from (4,) to (1, 4)
        if barcode_right.ndim == 1:
            barcode_right = barcode_right.reshape(1, 4)

    return barcode_left, barcode_right 

def convert_with_mst(prog, user_matrix, user_matrix_2, r1_mst_edge_idx, r1_mst_edge_w, num_vertices=-1):
    """
    Convert matrices with precomputed r1 MST.
    
    @param prog C library handle
    @param user_matrix First distance matrix (numpy array)
    @param user_matrix_2 Second distance matrix (numpy array)
    @param r1_mst_edge_idx r1 MST edges as numpy array (n-1, 2) - vertex indices
    @param r1_mst_edge_w r1 MST edge weights as numpy array (n-1,) - already sorted
    @param num_vertices Number of vertices
    @return barcode_left, barcode_right arrays
    """
    if isinstance(user_matrix, np.ndarray) and len(user_matrix) == 0:
        printHelpAndExit("Empty matrix provided")
        return
    
    num_rows, num_columns, num_entries, user_matrix_flat = distance_matrix_user_matrix(user_matrix)
    if user_matrix_flat is None:
        printHelpAndExit("Matrix was not created")
        return

    user_matrix = (ctypes.c_double * num_entries)(*user_matrix_flat)
    user_matrix_2 = (ctypes.c_double * num_entries)(*user_matrix_2.ravel())
    
    # Convert MST to C arrays
    # r1_mst_edge_idx: (n-1, 2) -> flat array [u0, v0, u1, v1, ...]
    n_edges = r1_mst_edge_idx.shape[0]
    r1_mst_idx_flat = (ctypes.c_int * (n_edges * 2))(*r1_mst_edge_idx.ravel().astype(np.int32))
    r1_mst_w_array = (ctypes.c_double * n_edges)(*r1_mst_edge_w.astype(np.float64))
    
    prog.run_matrix_with_mst.restype = RTD_Lite_result
    res = prog.run_matrix_with_mst(user_matrix, user_matrix_2, (ctypes.c_int64)(num_rows),
                                    r1_mst_idx_flat, r1_mst_w_array, (ctypes.c_bool)(True))
    
    # Ensure 2D arrays with shape (n, 4) even for empty or single barcode cases
    if res.left_bars == 0:
        barcode_left = np.empty((0, 4), dtype=np.int64)
    else:
        barcode_left = np.array([_bd_edges_to_array(res.left_to_right[NUM]) for NUM in range(res.left_bars)])
        if barcode_left.ndim == 1:
            if barcode_left.shape[0] == 4:
                barcode_left = barcode_left.reshape(1, 4)
            else:
                barcode_left = np.empty((0, 4), dtype=np.int64)
    
    if res.right_bars == 0:
        barcode_right = np.empty((0, 4), dtype=np.int64)
    else:
        barcode_right = np.array([_bd_edges_to_array(res.right_to_left[NUM]) for NUM in range(res.right_bars)])
        if barcode_right.ndim == 1:
            if barcode_right.shape[0] == 4:
                barcode_right = barcode_right.reshape(1, 4)
            else:
                barcode_right = np.empty((0, 4), dtype=np.int64)
    
    return barcode_left, barcode_right


def distance_matrix_user_matrix(user_matrix):
    if len(user_matrix.shape)!=2:
        printHelpAndExit("Matrix must be 2-dimensional")
        return

    num_rows, num_columns = user_matrix.shape

    num_entries= num_rows * num_columns

    return num_rows, num_columns, num_entries, user_matrix.ravel()