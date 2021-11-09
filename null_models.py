#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 13:22:11 2021

@author: mariapalazzi
"""
import random
import numpy as np
from scipy.optimize import root
from required_functions_null_models import avg_degrees_expoRG
#%%
def exponentialRG_proportional_proportional(M):
    '''
    Solve the system of equations of the maximum log-likelihood problem.
    The system of equations is solved using scipy's root function with
    the solver defined by 'method'. 
    The solutions correspond to the Lagrange multipliers. We used the Least-squares 
    method='lm' for the Levenberg-Marquardt approach.
    
    It can happen that the solver ``method`` used by ``scipy.root``
    does not converge to a solution.
    
    Payrato et al (PRX 2019) and F. Saracco et al (Sci Rep 2015)

    Parameters
    ----------
    M : array 
        the binary biadjacency matrix.

    Returns
    -------
    Pij : array, floats
        The probability matrix.
        Given the biadjacency matrix M, which describes the probabilities of
        row-node r and column-node c being linked.

    '''
    R,C=M.shape #rows and cols
    #degrees of "real" matrix
    cols_degr = M.sum(axis=0) 
    rows_degr = M.sum(axis=1)
    x0=np.random.uniform(0,1,size=(R+C))
    Pij = np.zeros((R,C));

    solution_vector =  root(avg_degrees_expoRG, x0,args=(rows_degr,cols_degr),method='lm')
    
    print("Solver successful for lagrange multipliers?", solution_vector.success)
    print(solution_vector.message)
    if solution_vector.success==False:
        Pij=Pij*np.nan
        print('Solution did not converge, returning P_ij with NAN')
        return Pij
    x = solution_vector.x[0:R]
    y = solution_vector.x[R:R+C]
    
    for p_index in range(len(rows_degr)):
        for a_index in range(len(cols_degr)):
            Pij[p_index,a_index] = (x[p_index]*y[a_index])/(1+x[p_index]*y[a_index])

    return Pij
#%%
def bascompte_probabilistic_proportional(M):
    '''
    Bascopmte's probabilistic null model. Bacompte PNAS 2003.

    Parameters
    ----------
    M : array
        the binary biadjacency matrix.

    Returns
    -------
    p_ij : array
        The probability matrix.
        Given the biadjacency matrix M, which describes the probabilities of
        row-node r and column-node c being linked.

    '''
    rows,cols=M.shape
    # dregrees of the rows and cols nodes
    cols_degr = M.sum(axis=0) 
    rows_degr = M.sum(axis=1)
    # normalized degrees
    rows_degr_norm = rows_degr/cols
    cols_degr_norm = cols_degr/rows

  #  M_null=np.zeros((rows,cols),dtype=int)
    M_n=np.zeros((rows,cols))
    
  #  M_rand_ij=np.random.uniform(0,1,size=(rows,cols))
    
    #obtaining the matrix of probabilities
    for i in range(rows):
        for j in range(cols):
            p_ij=0.5*(cols_degr_norm[j] + rows_degr_norm[i])
            M_n[i,j]=p_ij
    
    #null matrix
    #M_null=(M_n>=M_rand_ij).astype(int)
    return M_n
#%%
def corrected_probabilistic_proportional(M):
    '''
    corrected probabilistic null model, a variation of Bascompte's model

    Parameters
    ----------
    M : array
        the binary biadjacency matrix.

    Returns
    -------
    p_ij : array
        The probability matrix.
        Given the biadjacency matrix M, which describes the probabilities of
        row-node r and column-node c being linked.

    '''
    rows,cols=M.shape
    # dregrees of the rows and cols nodes
    cols_degr = M.sum(axis=0) 
    rows_degr = M.sum(axis=1)
    # normalized dregrees
    rows_degr_norm = rows_degr/cols
    cols_degr_norm = cols_degr/rows
    degr_rows_from_cols_sampling = (cols_degr_norm.sum())/cols
    degr_cols_from_rows_sampling = (rows_degr_norm.sum())/rows

    #M_null=np.zeros((rows,cols),dtype=int)
    M_n=np.zeros((rows,cols))
    
    #M_rand_ij=np.random.uniform(0,1,size=(rows,cols))
    
    #obtaining the matrix of probabilities
    for i in range(rows):
        for j in range(cols):
            p_ij=0.5*(cols_degr_norm[j] +
                                    ((1 - cols_degr_norm[j])*(rows_degr_norm[i] - degr_rows_from_cols_sampling)) +
                      rows_degr_norm[i] + 
                                    ((1 - rows_degr_norm[i])*(cols_degr_norm[j] - degr_cols_from_rows_sampling)) )
            if p_ij<0:
                p_ij=0
            M_n[i,j]=p_ij
    
    #null matrix
   # M_null=(M_n>=M_rand_ij).astype(int)
    # if symmetric==True:
    #     np.fill_diagonal(M_null, 0)
    #     M_=np.triu(M_null,k=1)+(np.triu(M_null,k=1)).T
    # else:
    #     M_=M_null
    return M_n
#%%
def equiprobable_equiprobable(matrix):
    '''
    Function to generate randomization with equiprobable degrees.
    
    Inputs:
    ----------
    input_matrix: array
        the binary biadjacency matrix
    
    output:
    ----------
    result: array
        randomized version of the original matrix with equiprobable degrees        
    '''
    matrix=np.array(matrix)	
    R,C=matrix.shape
    occs=matrix.sum()
    fill=occs/float(R*C)
    rm=np.zeros([R,C])
    while rm.sum()<occs:
        rr,rc=random.randrange(R),random.randrange(C)
        if random.random()<=fill:
            rm[rr][rc]=1
    return rm.astype(int)
#%%
def curve_ball(m,presences,num_iterations=-1):
    '''
    Function to generate randomization with fixed degrees.
    FF null model, Curveball (Strona et al. 2014)
    
    Inputs:
    ----------
    input_matrix: array
        the binary biadjacency matrix
    num_iterations: int
        Number of pair comparisons for each matrix generation, if empty, it takes 
        the smallets dimension times five.
    presences: list of lists (int)
        a list of list containing the indices of column (or rows) nodes to 
        wich each row (or column) node has a link
    
    output:
    ----------
    result: array
        randomized version of the original matrix with fixed degrees        
    '''
    r_hp=presences.copy()
    num_rows, num_cols = m.shape 
    num_iters = 5 * min(num_rows, num_cols) if num_iterations == -1 else num_iterations
    for rep in range(num_iters):
        ab = random.sample(range(len(r_hp)), 2) #randomly select two nodes
        a = ab[0]
        b = ab[1]
        ab = set(r_hp[a]) & set(r_hp[b])# overlap between the two nodes
        a_ba = set(r_hp[a]) - ab #other links of node a
        if len(a_ba) != 0:
            b_aa = set(r_hp[b]) - ab #other links of node b
            if len(b_aa) != 0:
                ab = list(ab)
                a_ba = list(a_ba)
                b_aa = list(b_aa)
                random.shuffle(a_ba)
                random.shuffle(b_aa)
                max_swap_extent=min(len(a_ba), len(b_aa)) #max value for pair extractions
                swap_extent = random.randint(1,max_swap_extent)
                #pair extractions
                r_hp[a] = ab+a_ba[:-swap_extent]+b_aa[-swap_extent:]
                r_hp[b] = ab+b_aa[:-swap_extent]+a_ba[-swap_extent:]
    out_mat = np.zeros([num_rows, num_cols], dtype='int8') if num_cols >= num_rows else np.zeros([num_cols,num_rows], dtype='int8')
    for r in range(min(num_rows, num_cols)):
        out_mat[r, r_hp[r]] = 1
        result = out_mat if num_cols >= num_rows else out_mat.T
    return result
