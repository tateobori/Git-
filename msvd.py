# -*- coding: utf-8 -*-

import scipy.linalg as spl
import numpy as np

def psvd(mat, rank):
    m,n = mat.shape
    rank = min(rank,m,n)
    try:
        u, s, vt = spl.svd(mat, full_matrices=False)
    except np.linalg.linalg.LinAlgError:
        print('svd did not converge')
        '''
        f = open('mat.txt','w');    f.write("np.array([")
        for i in range(mat.shape[0]):
            f.write("[{0:.20e}".format(mat[i][0]))
            for j in 1+np.arange(mat.shape[1]-1):
                f.write(",{0:.20e}".format(mat[i][j]))
            f.write("],")
        f.write("])\n");    f.flush()
        '''
        u, s, vt = spl.svd(mat, full_matrices=False,lapack_driver='gesvd')
    return u[:,:rank], s[:rank], vt[:rank,:]

def tensor_svd(a,axes0,axes1,rank):
    shape = np.array(a.shape)
    shape_row = [ shape[i] for i in axes0 ]
    shape_col = [ shape[i] for i in axes1 ]
    n_row = np.prod(shape_row)
    n_col = np.prod(shape_col)
    mat = np.reshape(np.transpose(a,axes0+axes1), (n_row,n_col))
    u, s, vt = psvd(mat, rank)
    return u.reshape(shape_row+[len(s)]), s, vt.reshape([len(s)]+shape_col)

def pQR(mat, rank):
    m,n = mat.shape
    rank = min(rank,m,n)
    try:
        q,r = spl.qr(mat)
    except np.linalg.linalg.LinAlgError:
        print('qr decomposiition error:',mat)
        #q,r = spl.qr(mat, full_matrices=False,lapack_driver='gesvd')
    return q[:,:rank], r[:rank,:]

def tensor_QR(a,axes0,axes1,rank):
    shape = np.array(a.shape)
    shape_row = [ shape[i] for i in axes0 ]
    shape_col = [ shape[i] for i in axes1 ]
    n_row = np.prod(shape_row)
    n_col = np.prod(shape_col)
    mat = np.reshape(np.transpose(a,axes0+axes1), (n_row,n_col))
    q,r = pQR(mat, rank)
    return q.reshape(shape_row+[q.shape[1]]), r.reshape([q.shape[1]]+shape_col)