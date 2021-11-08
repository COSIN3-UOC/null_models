#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 11:10:12 2021

@author: mariapalazzi
"""
import numpy as np
#%%
def avg_degrees_expoRG(x0,rows_degr,cols_degr):
    '''
    "The equations of the log-likelihood maximization problem to compute the 
    Lagrange multipliers resulting from the maximization 
    of the Shannon Gibs Entropy. Payrato et al (PRX 2019) and F. Saracco et al (Sci Rep 2015)

    Parameters
    ----------
    x0 : array, floats
        inital solution vector.
    rows_degr : array, ints
        degree sequence, rows.
    cols_degr : array, ints
        degree sequence, cols.

    Returns
    -------
    F : array, floats
        solution vector.

    '''
    R=len(rows_degr)
    C=len(cols_degr)
    x=x0[0:R]
    y=x0[R:R+C]
    
    v=np.zeros(R)
    h=np.zeros(C)

    for p_index in range(len(rows_degr)):
        v[p_index] = 0
        for a_index in range(len(cols_degr)):
            v[p_index] = v[p_index] + (x[p_index]*y[a_index]/(1+x[p_index]*y[a_index]))
    
    for a_index in range(len(cols_degr)):
        h[a_index] = 0
        for p_index in range(len(rows_degr)):
            h[a_index] = h[a_index] + (x[p_index]*y[a_index]/(1+x[p_index]*y[a_index]))

    F_row=v-rows_degr
    F_col=h-cols_degr
    F=np.hstack((F_row,F_col))
    return F
#%%
def find_presences(input_matrix):
    '''
    Function to identify the presences (links) in the biadjacency matrix 
    
    Inputs:
    ----------
    input_matrix: array
        the binary biadjacency matrix
    
    output:
    ----------
    hp: list of lists (int)
        a list of list containing the indices of column (or rows) nodes to 
        wich each row (or column) node has a link
        
    ''' 
    num_rows, num_cols = input_matrix.shape
    hp = []
    # to decide to which dimension iterate first 
    iters = num_rows if num_cols >= num_rows else num_cols 
    if num_cols >= num_rows:
        input_matrix_b = input_matrix
    else:
        input_matrix_b = input_matrix.T
    # creates a list of list containing the indices of column (or rows) nodes to 
    #wich each row (or column) node has a link
    for r in range(iters):
        hp.append(np.where(np.array(input_matrix_b[r]) == 1)[0])
    return hp
#%%
def z_scores(matrix_value,ensemble_values):
    '''
    Function to compute the z-scores to assess statistical significance 
    
    Inputs:
    ----------
    matrix_value: float
        the binary biadjacency matrix metric values (e.g. nestedness,modularity)
        ensemble_values: list  (float)
        a list containing the metric values of the null (random) matrices
    
    output:
    ----------
    zscore: float
        the z-score of real_value in the distribution of random_values
    ensemble_mean: float
        mean(ensemble_values)
    ensemble_std: float
        std(ensemble_values)
    
        
    '''
    ensemble_mean=np.mean(ensemble_values,dtype=np.float64)
    ensemble_std=np.std(ensemble_values,dtype=np.float64)
    z_score=(matrix_value-ensemble_mean)/(ensemble_std)
    return z_score,ensemble_mean,ensemble_std