# -*- coding: utf-8 -*-

#import scipy as sp
import scipy.linalg as spl
import numpy as np
import CTMRG2x2 as CTMRG
from msvd import tensor_svd

################################################################
def initial_iPEPS(D):
    ## random
    a = np.random.random((D,D,D,D,2)) +1.0j
    b = np.random.random((D,D,D,D,2)) -1.0j
    ## ferro
    #a = np.array([1.,0.])[None,None,None,None,:];b=a
    ## antiferro
    #a = np.array([1.,0.])[None,None,None,None,:]
    #b = np.array([0.,1.])[None,None,None,None,:]
    
    # vector lu, lr, ld, ll
    l = np.ones(a.shape[0], dtype=float)
    for i in np.arange(len(l)):    l[i] /= 10**i
    l /= np.sqrt(np.dot(l,l))
    
    return a, b, l, l, l, l

#### Simple Update   ###################################################

def SimpleUpdate(a,b,lu,lr,ld,ll,U,D):
    #    /   /
    # --a---b--
    #  /|  /|
    a = a*lu[:,None,None,None,None]*lr[None,:,None,None,None]\
        *ld[None,None,:,None,None]*ll[None,None,None,:,None]
    b = b*ld[:,None,None,None,None]*ll[None,:,None,None,None]\
        *lu[None,None,:,None,None]
    tmp = np.tensordot(np.tensordot(a,b,(1,3)),U,((3,7),(0,1)))
    u, s, vdag = tensor_svd(tmp,(0,1,2,6),(3,4,5,7),D)
    a = u.transpose((0,4,1,2,3))*(1/lu)[:,None,None,None,None]\
        *(1/ld)[None,None,:,None,None]*(1/ll)[None,None,None,:,None]
    b = vdag.transpose((1,2,3,0,4))*(1/ld)[:,None,None,None,None]\
        *(1/ll)[None,:,None,None,None]*(1/lu)[None,None,:,None,None]
    lr = s/np.sqrt(np.dot(s,s))
    #      /
    #   --a--
    #    /|
    # --b--
    #  /|
    a = a*lu[:,None,None,None,None]*lr[None,:,None,None,None]\
        *ld[None,None,:,None,None]*ll[None,None,None,:,None]
    b = b*ll[None,:,None,None,None]*lu[None,None,:,None,None]\
        *lr[None,None,None,:,None]
    tmp = np.tensordot(np.tensordot(a,b,(2,0)),U,((3,7),(0,1)))
    u, s, vdag = tensor_svd(tmp,(0,1,2,6),(3,4,5,7),D)
    a = u.transpose((0,1,4,2,3))*(1/lu)[:,None,None,None,None]\
        *(1/lr)[None,:,None,None,None]*(1/ll)[None,None,None,:,None]
    b = vdag*(1/ll)[None,:,None,None,None]\
        *(1/lu)[None,None,:,None,None]*(1/lr)[None,None,None,:,None]
    ld = s/np.sqrt(np.dot(s,s))
    #    /   /
    # --b---a--
    #  /|  /|
    a = a*lu[:,None,None,None,None]*lr[None,:,None,None,None]\
        *ld[None,None,:,None,None]*ll[None,None,None,:,None]
    b = b*ld[:,None,None,None,None]*lu[None,None,:,None,None]\
        *lr[None,None,None,:,None]
    tmp = np.tensordot(np.tensordot(a,b,(3,1)),U,((3,7),(0,1)))
    u, s, vdag = tensor_svd(tmp,(0,1,2,6),(3,4,5,7),D)
    a = u.transpose((0,1,2,4,3))*(1/lu)[:,None,None,None,None]\
        *(1/lr)[None,:,None,None,None]*(1/ld)[None,None,:,None,None]
    b = vdag.transpose((1,0,2,3,4))*(1/ld)[:,None,None,None,None]\
        *(1/lu)[None,None,:,None,None]*(1/lr)[None,None,None,:,None]
    ll = s/np.sqrt(np.dot(s,s))
    #      /
    #   --b--
    #    /|
    # --a--
    #  /|
    a = a*lu[:,None,None,None,None]*lr[None,:,None,None,None]\
        *ld[None,None,:,None,None]*ll[None,None,None,:,None]
    b = b*ld[:,None,None,None,None]*ll[None,:,None,None,None]\
        *lr[None,None,None,:,None]
    tmp = np.tensordot(np.tensordot(a,b,(0,2)),U,((3,7),(0,1)))
    u, s, vdag = tensor_svd(tmp,(0,1,2,6),(3,4,5,7),D)
    a = u.transpose((4,0,1,2,3))*(1/lr)[None,:,None,None,None]\
        *(1/ld)[None,None,:,None,None]*(1/ll)[None,None,None,:,None]
    b = vdag.transpose((1,2,0,3,4))*(1/ld)[:,None,None,None,None]\
        *(1/ll)[None,:,None,None,None]*(1/lr)[None,None,None,:,None]
    lu = s/np.sqrt(np.dot(s,s))

    return a,b,lu,lr,ld,ll


# Cb--Tb1-Tb2-Cc
# |   |   |   |
# Ta2-a= =b---Tc1
# |   |   |   |      : このネットワークにおけるaとbを更新する
# Ta1-B---A---Tc2
# |   |   |   |
# Ca--Td2-Td1-Cd

#### 正方格子

def SimpleUpdate_abcd(a,b,c,d,l1,l2,l3,l4,l5,l6,l7,l8,U,D):
    """
      |  |
    --a--b--
      |  |
    --c--d-- 
      |  |
    """
    ###############################################################
    #    /   /
    # --a---b--
    #  /|  /|
    a = a*l1[:,None,None,None,None]*l2[None,:,None,None,None]\
        *l3[None,None,:,None,None]*l4[None,None,None,:,None]
    b = b*l7[:,None,None,None,None]*l4[None,:,None,None,None]\
        *l5[None,None,:,None,None]
    tmp = np.tensordot(np.tensordot(a,b,(1,3)),U,((3,7),(0,1)))
    u, s, vdag = tensor_svd(tmp,(0,1,2,6),(3,4,5,7),D)
    a = u.transpose((0,4,1,2,3))*(1/l1)[:,None,None,None,None]\
        *(1/l3)[None,None,:,None,None]*(1/l4)[None,None,None,:,None]
    b = vdag.transpose((1,2,3,0,4))*(1/l7)[:,None,None,None,None]\
        *(1/l4)[None,:,None,None,None]*(1/l5)[None,None,:,None,None]
    l2 = s/np.sqrt(np.dot(s,s))
    #      /
    #   --a--
    #    /|
    # --c--
    #  /|
    a = a*l1[:,None,None,None,None]*l2[None,:,None,None,None]\
        *l3[None,None,:,None,None]*l4[None,None,None,:,None]
    c = c*l8[None,:,None,None,None]*l1[None,None,:,None,None]\
        *l6[None,None,None,:,None]
    tmp = np.tensordot(np.tensordot(a,c,(2,0)),U,((3,7),(0,1)))
    u, s, vdag = tensor_svd(tmp,(0,1,2,6),(3,4,5,7),D)
    a = u.transpose((0,1,4,2,3))*(1/l1)[:,None,None,None,None]\
        *(1/l2)[None,:,None,None,None]*(1/l4)[None,None,None,:,None]
    c = vdag*(1/l8)[None,:,None,None,None]\
        *(1/l1)[None,None,:,None,None]*(1/l6)[None,None,None,:,None]
    l3 = s/np.sqrt(np.dot(s,s))
    #    /   /
    # --b---a--
    #  /|  /|
    a = a*l1[:,None,None,None,None]*l2[None,:,None,None,None]\
        *l3[None,None,:,None,None]*l4[None,None,None,:,None]
    b = b*l7[:,None,None,None,None]*l5[None,None,:,None,None]\
        *l2[None,None,None,:,None]
    tmp = np.tensordot(np.tensordot(a,b,(3,1)),U,((3,7),(0,1)))
    u, s, vdag = tensor_svd(tmp,(0,1,2,6),(3,4,5,7),D)
    a = u.transpose((0,1,2,4,3))*(1/l1)[:,None,None,None,None]\
        *(1/l2)[None,:,None,None,None]*(1/l3)[None,None,:,None,None]
    b = vdag.transpose((1,0,2,3,4))*(1/l7)[:,None,None,None,None]\
        *(1/l5)[None,None,:,None,None]*(1/l2)[None,None,None,:,None]
    l4 = s/np.sqrt(np.dot(s,s))
    #      /
    #   --c--
    #    /|
    # --a--
    #  /|
    a = a*l1[:,None,None,None,None]*l2[None,:,None,None,None]\
        *l3[None,None,:,None,None]*l4[None,None,None,:,None]
    c = c*l3[:,None,None,None,None]*l8[None,:,None,None,None]\
        *l6[None,None,None,:,None]
    tmp = np.tensordot(np.tensordot(a,c,(0,2)),U,((3,7),(0,1)))
    u, s, vdag = tensor_svd(tmp,(0,1,2,6),(3,4,5,7),D)
    a = u.transpose((4,0,1,2,3))*(1/l2)[None,:,None,None,None]\
        *(1/l3)[None,None,:,None,None]*(1/l4)[None,None,None,:,None]
    c = vdag.transpose((1,2,0,3,4))*(1/l3)[:,None,None,None,None]\
        *(1/l8)[None,:,None,None,None]*(1/l6)[None,None,None,:,None]
    l1 = s/np.sqrt(np.dot(s,s))

    #####################################################################
    #    /   /
    # --d---c--
    #  /|  /|
    d = d*l5[:,None,None,None,None]*l6[None,:,None,None,None]\
        *l7[None,None,:,None,None]*l8[None,None,None,:,None]
    c = c*l3[:,None,None,None,None]*l8[None,:,None,None,None]\
        *l1[None,None,:,None,None]
    tmp = np.tensordot(np.tensordot(d,c,(1,3)),U,((3,7),(0,1)))
    u, s, vdag = tensor_svd(tmp,(0,1,2,6),(3,4,5,7),D)
    d = u.transpose((0,4,1,2,3))*(1/l5)[:,None,None,None,None]\
        *(1/l7)[None,None,:,None,None]*(1/l8)[None,None,None,:,None]
    c = vdag.transpose((1,2,3,0,4))*(1/l3)[:,None,None,None,None]\
        *(1/l8)[None,:,None,None,None]*(1/l1)[None,None,:,None,None]
    l6 = s/np.sqrt(np.dot(s,s))
    #      /
    #   --d--
    #    /|
    # --b--
    #  /|
    d = d*l5[:,None,None,None,None]*l6[None,:,None,None,None]\
        *l7[None,None,:,None,None]*l8[None,None,None,:,None]
    b = b*l4[None,:,None,None,None]*l5[None,None,:,None,None]\
        *l2[None,None,None,:,None]
    tmp = np.tensordot(np.tensordot(d,b,(2,0)),U,((3,7),(0,1)))
    u, s, vdag = tensor_svd(tmp,(0,1,2,6),(3,4,5,7),D)
    d = u.transpose((0,1,4,2,3))*(1/l5)[:,None,None,None,None]\
        *(1/l6)[None,:,None,None,None]*(1/l8)[None,None,None,:,None]
    b = vdag*(1/l4)[None,:,None,None,None]\
        *(1/l5)[None,None,:,None,None]*(1/l2)[None,None,None,:,None]
    l7= s/np.sqrt(np.dot(s,s))
    #    /   /
    # --c---d--
    #  /|  /|
    d = d*l5[:,None,None,None,None]*l6[None,:,None,None,None]\
        *l7[None,None,:,None,None]*l8[None,None,None,:,None]
    c = c*l3[:,None,None,None,None]*l1[None,None,:,None,None]\
        *l6[None,None,None,:,None]
    tmp = np.tensordot(np.tensordot(d,c,(3,1)),U,((3,7),(0,1)))
    u, s, vdag = tensor_svd(tmp,(0,1,2,6),(3,4,5,7),D)
    d = u.transpose((0,1,2,4,3))*(1/l5)[:,None,None,None,None]\
        *(1/l6)[None,:,None,None,None]*(1/l7)[None,None,:,None,None]
    c = vdag.transpose((1,0,2,3,4))*(1/l3)[:,None,None,None,None]\
        *(1/l1)[None,None,:,None,None]*(1/l6)[None,None,None,:,None]
    l8 = s/np.sqrt(np.dot(s,s))
    #      /
    #   --b--
    #    /|
    # --d--
    #  /|
    d = d*l5[:,None,None,None,None]*l6[None,:,None,None,None]\
        *l7[None,None,:,None,None]*l8[None,None,None,:,None]
    b = b*l7[:,None,None,None,None]*l4[None,:,None,None,None]\
        *l2[None,None,None,:,None]
    tmp = np.tensordot(np.tensordot(d,b,(0,2)),U,((3,7),(0,1)))
    u, s, vdag = tensor_svd(tmp,(0,1,2,6),(3,4,5,7),D)
    d = u.transpose((4,0,1,2,3))*(1/l6)[None,:,None,None,None]\
        *(1/l7)[None,None,:,None,None]*(1/l8)[None,None,None,:,None]
    b = vdag.transpose((1,2,0,3,4))*(1/l7)[:,None,None,None,None]\
        *(1/l4)[None,:,None,None,None]*(1/l2)[None,None,None,:,None]
    l5 = s/np.sqrt(np.dot(s,s))

    return a,b,c,d,l1,l2,l3,l4,l5,l6,l7,l8

def SimpleUpdate_abcd_NN(a,b,c,d,l1,l2,l3,l4,l5,l6,l7,l8,U,D):
    ###############################################################
    #       /   /
    #    --a---b--
    #     /|  /|
    #       -d-
    #        |
    #  a  -->  b --> d の順に時計回りの添字
    a = a*l1[:,None,None,None,None]*l2[None,:,None,None,None]\
        *l3[None,None,:,None,None]*l4[None,None,None,:,None]

    b = b*l7[:,None,None,None,None]*l4[None,:,None,None,None]
    d = d*l5[:,None,None,None,None]*l6[None,:,None,None,None]\
        *l7[None,None,:,None,None]*l8[None,None,None,:,None]
   #print(a.shape, b.shape, c.shape, d.shape)
    tmp = np.transpose(
        np.tensordot(
            U, np.tensordot(
                a, np.tensordot(
                    b, d, ([2], [0])
                ), ([1], [2])
            ), ([0, 1], [3, 10])
        ), [2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 1]
    )
    u,s,vdag = tensor_svd(tmp, (0,1,2,3),(4,5,6,7,8,9,10),D)
    a = u*(1./l1)[:,None,None,None,None]*(1./l3)[None,:,None,None,None]\
         *(1./l4)[None,None,:,None,None]
    a=a.transpose(0,4,1,2,3) ## aを求めた
    l2 = s/np.sqrt(np.dot(s,s))
    vdag=vdag*l2[:,None,None,None,None,None,None,None]
    u,s,vdag = tensor_svd(vdag, (0,1,2,3),(4,5,6,7),D)
    l5 = s/np.sqrt(np.dot(s,s))
    b= (u*(1./l7)[None,:,None,None,None]*(1./l4)[None,None,:,None,None]).\
         transpose(1,2,4,0,3)
    d=vdag*(1./l6)[None,:,None,None,None]*(1./l7)[None,None,:,None,None]\
        *(1./l8)[None,None,None,:,None]
    ###############################################################
    
    #       /   /
    #    --a---b--
    #     /|  /|
    #  --c--
    #   /|    
    #  b  -->  a --> c の順に時計回りの添字

    a = a*l1[:,None,None,None,None]*l2[None,:,None,None,None]\
        *l3[None,None,:,None,None]*l4[None,None,None,:,None]
    b = b*l7[:,None,None,None,None]*l4[None,:,None,None,None]\
        *l5[None,None,:,None,None]
    c = c*l8[None,:,None,None,None]*l1[None,None,:,None,None]\
        *l6[None,None,None,:,None]

    tmp = np.transpose(
        np.tensordot(
            U, np.tensordot(
                b, np.tensordot(
                    a, c, ([2], [0])
                ), ([3], [1])
            ), ([0, 1], [3, 10])
        ), [2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 1]
    )
    u,s,vdag = tensor_svd(tmp, (0,1,2,3),(4,5,6,7,8,9,10),D)
    b= (u*(1./l7)[:,None,None,None,None]*(1/l4)[None,:,None,None,None]*\
        (1./l5)[None,None,:,None,None]).transpose(0,1,2,4,3)

    l2 = s/np.sqrt(np.dot(s,s))
    vdag=vdag*l2[:,None,None,None,None,None,None,None]
    u,s,vdag = tensor_svd(vdag, (0,1,2,3),(4,5,6,7),D)
    l3 = s/np.sqrt(np.dot(s,s))

    a = (u*(1./l1)[None,:,None,None,None]*(1./l4)[None,None,:,None,None]).\
        transpose(1,0,4,2,3)
    c = vdag*(1./l8)[None,:,None,None,None]*(1./l1)[None,None,:,None,None]\
        *(1./l6)[None,None,None,:,None]

    ###############################################################
    #       /   
    #    --a--
    #     /|   /
    #  --c----d-
    #   /|   /|
    #  a  -->  c --> d の順に時計回りの添字

    a = a*l1[:,None,None,None,None]*l2[None,:,None,None,None]\
        *l3[None,None,:,None,None]*l4[None,None,None,:,None]
    c = c*l1[None,None,:,None,None]*l6[None,None,None,:,None]
    d = d*l5[:,None,None,None,None]*l6[None,:,None,None,None]\
        *l7[None,None,:,None,None]*l8[None,None,None,:,None]

    tmp = np.transpose(
        np.tensordot(
            U, np.tensordot(
                a, np.tensordot(
                    c, d, ([1], [3])
                ), ([2], [0])
            ), ([0, 1], [3, 10])
        ), [2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 1]
    )
    u,s,vdag = tensor_svd(tmp, (0,1,2,3),(4,5,6,7,8,9,10),D)
    a=(u*(1./l1)[:,None,None,None,None]*(1./l2)[None,:,None,None,None]\
        *(1./l4)[None,None,:,None,None]).transpose(0,1,4,2,3)
    l3 = s/np.sqrt(np.dot(s,s))
    vdag=vdag*l3[:,None,None,None,None,None,None,None]
    u,s,vdag = tensor_svd(vdag, (0,1,2,3),(4,5,6,7),D)
    l8 = s/np.sqrt(np.dot(s,s))

    c = (u*(1./l1)[None,:,None,None,None]*(1./l6)[None,None,:,None,None]\
        ).transpose(0,4,1,2,3)
    d = (vdag*(1./l5)[None,:,None,None,None]*(1./l6)[None,None,:,None,None]\
        *(1./l7)[None,None,None,:,None]).transpose(1,2,3,0,4)

    ###############################################################
    #            /   
    #         --b--
    #     /    /|
    #  --c----d-
    #   /|   /|
    #  b  -->  d --> c の順に時計回りの添字
    b = b*l7[:,None,None,None,None]*l4[None,:,None,None,None]\
        *l2[None,None,None,:,None]
    c = c*l3[:,None,None,None,None]*l1[None,None,:,None,None]\
        *l6[None,None,None,:,None]
    d = d*l5[:,None,None,None,None]*l6[None,:,None,None,None]\
        *l7[None,None,:,None,None]*l8[None,None,None,:,None]
    tmp = np.transpose(
        np.tensordot(
            U, np.tensordot(
                b, np.tensordot(
                    d, c, ([3], [1])
                ), ([2], [0])
            ), ([0, 1], [10, 3])
        ), [2, 3, 4, 1, 5, 6, 7, 8, 9, 10, 0]
    )

    u,s,vdag = tensor_svd(tmp, (0,1,2,3),(4,5,6,7,8,9,10),D)
    l5 = s/np.sqrt(np.dot(s,s))
    vdag=vdag*l5[:,None,None,None,None,None,None,None]
    b = (u*(1./l7)[:,None,None,None,None]*(1./l4)[None,:,None,None,None]\
        *(1./l2)[None,None,:,None,None]).transpose(0,1,4,2,3)
    u,s,vdag = tensor_svd(vdag, (0,1,2,3),(4,5,6,7),D)

    d = (u*(1./l6)[None,:,None,None,None]*(1./l7)[None,None,:,None,None]\
        ).transpose(0,1,2,4,3)
    c = (vdag*(1./l3)[None,:,None,None,None]*(1./l1)[None,None,:,None,None]\
        *(1./l6)[None,None,:,None,None]).transpose(1,0,2,3,4)

    
    return a,b,c,d,l1,l2,l3,l4,l5,l6,l7,l8

#### 三角格子
## Energy per site = -0.546 
## m = 0.30 
def SimpleUpdate_abcd_triangular_right_upper(a,c,b,l1,l2,l3,l4,l5,l6,U,D):
    ###############################################################
    
    
    #       /   /
    #    --a---c--
    #     /|  /|
    #       -b-
    #       /|
    #  a  -->  c --> b の順に時計回りの添字
    a = a*l1[:,None,None,None,None]*l2[None,:,None,None,None]\
        *l3[None,None,:,None,None]*l4[None,None,None,:,None]

    c = c*l3[:,None,None,None,None]*l5[None,:,None,None,None]
    b = b*l6[:,None,None,None,None]*l4[None,:,None,None,None]\
        *l1[None,None,:,None,None]*l5[None,None,None,:,None]
   #print(a.shape, b.shape, c.shape, d.shape)
    tmp = np.transpose(
        np.tensordot(
            np.tensordot(
                U, np.tensordot(
                    U, U, ([3], [1])
                ), ([2, 3], [3, 0])
            ), np.tensordot(
                a, np.tensordot(
                    c, b, ([2], [0])
                ), ([1], [2])
            ), ([0, 1, 2], [3, 6, 10])
        ), [3, 4, 5, 1, 6, 7, 0, 8, 9, 10, 2]
    )
    u,s,vdag = tensor_svd(tmp, (0,1,2,3),(4,5,6,7,8,9,10),D)
    a = u*(1./l1)[:,None,None,None,None]*(1./l3)[None,:,None,None,None]\
         *(1./l4)[None,None,:,None,None]
    a=a.transpose(0,4,1,2,3) ## aを求めた
    l2 = s/np.sqrt(np.dot(s,s))
    vdag=vdag*l2[:,None,None,None,None,None,None,None]
    u,s,vdag = tensor_svd(vdag, (0,1,2,3),(4,5,6,7),D)
    l6 = s/np.sqrt(np.dot(s,s))
    c = (u*(1./l3)[None,:,None,None,None]*(1./l5)[None,None,:,None,None]).\
         transpose(1,2,4,0,3)
    b = vdag*(1./l4)[None,:,None,None,None]*(1./l1)[None,None,:,None,None]\
        *(1./l5)[None,None,None,:,None]
    
    """
    ###############################################################
    
    #       /   
    #    --a--
    #     /|   /
    #  --c----b-
    #   /|   /|
    #  a  -->  c --> b の順に時計回りの添字

    a = a*l1[:,None,None,None,None]*l2[None,:,None,None,None]\
        *l3[None,None,:,None,None]*l4[None,None,None,:,None]
    c = c*l6[None,None,:,None,None]*l2[None,None,None,:,None]
    b = b*l6[:,None,None,None,None]*l4[None,:,None,None,None]\
        *l1[None,None,:,None,None]*l5[None,None,None,:,None]

    tmp = np.transpose(
        np.tensordot(
            np.tensordot(
                U, np.tensordot(
                    U, U, ([3], [1])
                ), ([2, 3], [3, 0])
            ), np.tensordot(
                a, np.tensordot(
                    c, b, ([1], [3])
                ), ([2], [0])
            ), ([0, 1, 2], [3, 6, 10])
        ), [3, 4, 5, 1, 6, 7, 0, 8, 9, 10, 2]
    )
    u,s,vdag = tensor_svd(tmp, (0,1,2,3),(4,5,6,7,8,9,10),D)
    a=(u*(1./l1)[:,None,None,None,None]*(1./l2)[None,:,None,None,None]\
        *(1./l4)[None,None,:,None,None]).transpose(0,1,4,2,3)
    l3 = s/np.sqrt(np.dot(s,s))
    vdag=vdag*l3[:,None,None,None,None,None,None,None]
    u,s,vdag = tensor_svd(vdag, (0,1,2,3),(4,5,6,7),D)
    l5 = s/np.sqrt(np.dot(s,s))

    c = (u*(1./l6)[None,:,None,None,None]*(1./l2)[None,None,:,None,None]\
        ).transpose(0,4,1,2,3)
    b = (vdag*(1./l6)[None,:,None,None,None]*(1./l4)[None,None,:,None,None]\
        *(1./l1)[None,None,None,:,None]).transpose(1,2,3,0,4)
    """
    return a,c,b,l1,l2,l3,l4,l5,l6

def SimpleUpdate_abcd_triangular_left_lowwer(a,c,b,l1,l2,l3,l4,l5,l6,U,D):
    ###############################################################
    
    """
    #       /   /
    #    --a---c--
    #     /|  /|
    #       -b-
    #       /|
    #  a  -->  c --> b の順に時計回りの添字
    a = a*l1[:,None,None,None,None]*l2[None,:,None,None,None]\
        *l3[None,None,:,None,None]*l4[None,None,None,:,None]

    c = c*l3[:,None,None,None,None]*l5[None,:,None,None,None]
    b = b*l6[:,None,None,None,None]*l4[None,:,None,None,None]\
        *l1[None,None,:,None,None]*l5[None,None,None,:,None]
   #print(a.shape, b.shape, c.shape, d.shape)
    tmp = np.transpose(
        np.tensordot(
            np.tensordot(
                U, np.tensordot(
                    U, U, ([3], [1])
                ), ([2, 3], [3, 0])
            ), np.tensordot(
                a, np.tensordot(
                    c, b, ([2], [0])
                ), ([1], [2])
            ), ([0, 1, 2], [3, 6, 10])
        ), [3, 4, 5, 1, 6, 7, 0, 8, 9, 10, 2]
    )
    u,s,vdag = tensor_svd(tmp, (0,1,2,3),(4,5,6,7,8,9,10),D)
    a = u*(1./l1)[:,None,None,None,None]*(1./l3)[None,:,None,None,None]\
         *(1./l4)[None,None,:,None,None]
    a=a.transpose(0,4,1,2,3) ## aを求めた
    l2 = s/np.sqrt(np.dot(s,s))
    vdag=vdag*l2[:,None,None,None,None,None,None,None]
    u,s,vdag = tensor_svd(vdag, (0,1,2,3),(4,5,6,7),D)
    l6 = s/np.sqrt(np.dot(s,s))
    c = (u*(1./l3)[None,:,None,None,None]*(1./l5)[None,None,:,None,None]).\
         transpose(1,2,4,0,3)
    b = vdag*(1./l4)[None,:,None,None,None]*(1./l1)[None,None,:,None,None]\
        *(1./l5)[None,None,None,:,None]
    
    """
    ###############################################################
    
    #       /   
    #    --a--
    #     /|   /
    #  --c----b-
    #   /|   /|
    #  a  -->  c --> b の順に時計回りの添字

    a = a*l1[:,None,None,None,None]*l2[None,:,None,None,None]\
        *l3[None,None,:,None,None]*l4[None,None,None,:,None]
    c = c*l6[None,None,:,None,None]*l2[None,None,None,:,None]
    b = b*l6[:,None,None,None,None]*l4[None,:,None,None,None]\
        *l1[None,None,:,None,None]*l5[None,None,None,:,None]

    tmp = np.transpose(
        np.tensordot(
            np.tensordot(
                U, np.tensordot(
                    U, U, ([3], [1])
                ), ([2, 3], [3, 0])
            ), np.tensordot(
                a, np.tensordot(
                    c, b, ([1], [3])
                ), ([2], [0])
            ), ([0, 1, 2], [3, 6, 10])
        ), [3, 4, 5, 1, 6, 7, 0, 8, 9, 10, 2]
    )
    u,s,vdag = tensor_svd(tmp, (0,1,2,3),(4,5,6,7,8,9,10),D)
    a=(u*(1./l1)[:,None,None,None,None]*(1./l2)[None,:,None,None,None]\
        *(1./l4)[None,None,:,None,None]).transpose(0,1,4,2,3)
    l3 = s/np.sqrt(np.dot(s,s))
    vdag=vdag*l3[:,None,None,None,None,None,None,None]
    u,s,vdag = tensor_svd(vdag, (0,1,2,3),(4,5,6,7),D)
    l5 = s/np.sqrt(np.dot(s,s))

    c = (u*(1./l6)[None,:,None,None,None]*(1./l2)[None,None,:,None,None]\
        ).transpose(0,4,1,2,3)
    b = (vdag*(1./l6)[None,:,None,None,None]*(1./l4)[None,None,:,None,None]\
        *(1./l1)[None,None,None,:,None]).transpose(1,2,3,0,4)
    
    return a,c,b,l1,l2,l3,l4,l5,l6

def SimpleUpdate_triangular(a,c,b,l1,l2,l3,l4,l5,l6,U,U1,U2,D):

    
    #       /   
    #    --a--
    #     /|   /
    #  --c----b-
    #   /|   /|
    #  a  -->  c --> b の順に時計回りの添字
    # acをupdateしてからcbをupdateする

    ## 垂直, 水平方向のupdate

    #      /
    #   --a--
    #    /|
    # --c--
    #  /|

    a = a*l1[:,None,None,None,None]*l2[None,:,None,None,None]\
        *l3[None,None,:,None,None]*l4[None,None,None,:,None]
    c = c*l5[None,:,None,None,None]*l6[None,None,:,None,None]\
        *l2[None,None,None,:,None]

    tmp = np.tensordot(np.tensordot(a,c,(2,0)),U,((3,7),(2,3)))
    u, s, vdag = tensor_svd(tmp,(0,1,2,6),(3,4,5,7),D)
    a = u.transpose((0,1,4,2,3))*(1/l1)[:,None,None,None,None]\
        *(1/l2)[None,:,None,None,None]*(1/l4)[None,None,None,:,None]
    c = vdag*(1/l5)[None,:,None,None,None]\
        *(1/l6)[None,None,:,None,None]*(1/l2)[None,None,None,:,None]
    l3 = s/np.sqrt(np.dot(s,s))

    #    /   /
    # --c---b--
    #  /|  /|
    c = c*l3[:,None,None,None,None]*l5[None,:,None,None,None]\
        *l6[None,None,:,None,None]*l2[None,None,None,:,None]
    b = b*l6[:,None,None,None,None]*l4[None,:,None,None,None]\
        *l1[None,None,:,None,None]

    tmp = np.tensordot(np.tensordot(c,b,(1,3)),U,((3,7),(2,3)))
    u, s, vdag = tensor_svd(tmp,(0,1,2,6),(3,4,5,7),D)
    c = u.transpose((0,4,1,2,3))*(1/l3)[:,None,None,None,None]\
        *(1/l6)[None,None,:,None,None]*(1/l2)[None,None,None,:,None]
    b = vdag.transpose((1,2,3,0,4))*(1/l6)[:,None,None,None,None]\
        *(1/l4)[None,:,None,None,None]*(1/l1)[None,None,:,None,None]
    l5 = s/np.sqrt(np.dot(s,s))

    ## 斜め方向のupdate
    #      /
    #   --a--
    #    /|
    # --b--
    #  /|

    a = a*l1[:,None,None,None,None]*l2[None,:,None,None,None]\
        *l3[None,None,:,None,None]*l4[None,None,None,:,None]
    c = c*l5[None,:,None,None,None]*l6[None,None,:,None,None]\
        *l2[None,None,None,:,None]

    tmp = np.tensordot(np.tensordot(a,c,(2,0)),U1,((3,7),(2,3)))
    u, s, vdag = tensor_svd(tmp,(0,1,2,6),(3,4,5,7),D)
    a = u.transpose((0,1,4,2,3))*(1/l1)[:,None,None,None,None]\
        *(1/l2)[None,:,None,None,None]*(1/l4)[None,None,None,:,None]
    c = vdag*(1/l5)[None,:,None,None,None]\
        *(1/l6)[None,None,:,None,None]*(1/l2)[None,None,None,:,None]
    l3 = s/np.sqrt(np.dot(s,s))


    #    /   /
    # --c---b--
    #  /|  /|
    c = c*l3[:,None,None,None,None]*l5[None,:,None,None,None]\
        *l6[None,None,:,None,None]*l2[None,None,None,:,None]
    b = b*l6[:,None,None,None,None]*l4[None,:,None,None,None]\
        *l1[None,None,:,None,None]

    tmp = np.tensordot(np.tensordot(c,b,(1,3)),U2,((3,7),(2,3)))
    u, s, vdag = tensor_svd(tmp,(0,1,2,6),(3,4,5,7),D)
    c = u.transpose((0,4,1,2,3))*(1/l3)[:,None,None,None,None]\
        *(1/l6)[None,None,:,None,None]*(1/l2)[None,None,None,:,None]
    b = vdag.transpose((1,2,3,0,4))*(1/l6)[:,None,None,None,None]\
        *(1/l4)[None,:,None,None,None]*(1/l1)[None,None,:,None,None]
    l5 = s/np.sqrt(np.dot(s,s))
    
    return a,c,b,l3,l5

#### Full Update   ###################################################

def FullUpdate(a,b,Ca,Ta1,Ta2,Cb,Tb1,Tb2,Cc,Tc1,Tc2,Cd,Td1,Td2,U,D):
    # create reduced tensors
    #      0           0
    #     /           /
    # 3--a--1  =  1--X-3---0-aR--1
    #   /|          /        |
    #  2 4         2         2
    tmp = a.transpose([0,3,2,1,4]);  sh = tmp.shape
    X, aR = spl.qr(tmp.reshape(np.prod(sh[:3]),sh[3]*sh[4]),
                   mode='economic')
    X = X.reshape(sh[:3]+(sh[3]*sh[4],))
    aR = aR.reshape((sh[3]*sh[4],)+sh[3:])
    #      0                    0
    #     /                    /
    # 3--b--1  =  1--bL-0---3-Y--1
    #   /|           |       /
    #  2 4           2      2
    sh = b.shape
    Y, bL = spl.qr(b.reshape(np.prod(sh[:3]),sh[3]*sh[4]),mode='economic')
    Y = Y.reshape(sh[:3]+(sh[3]*sh[4],))
    bL = bL.reshape((sh[3]*sh[4],)+sh[3:])
    
    # create an environment N_{LR}
    print (X.shape, Y.shape)
    NLR = CTMRG.CreateNLR(Ca,Ta1,Ta2,Cb,Tb1,Tb2,Cc,Tc1,Tc2,Cd,Td1,Td2,X,Y,
                          CTMRG.ContractPhysBond(b,np.conj(b)),
                          CTMRG.ContractPhysBond(a,np.conj(a)))
    #NLR = GaugeFix(NLR)
    
    # update aR in the fixed bL
    tmp = np.tensordot(np.tensordot(aR,bL,(1,1)),U,((1,3),(0,1)))
    _, sv, bLtil = tensor_svd(tmp,(0,2),(1,3),D)
    # bLtilの初期値はUを作用させてSVDしたもの
    bLtil = (bLtil*np.sqrt(sv)[:,None,None]).transpose((1,0,2))
    precost = CTMRG.costfunction(aR,bL,NLR,tmp)
    while 1:
        R = np.tensordot(np.tensordot(NLR,bLtil,(2,0)),np.conj(bLtil),
                         ((2,4),(0,2)))
        S = np.tensordot(np.tensordot(NLR,tmp,((0,2),(0,1))),
                         np.conj(bLtil),((1,3),(0,2)))
        aRtil = spl.solve(R.transpose((1,3,0,2)).reshape((2*D*D,2*D*D)),
                          S.transpose((0,2,1)).reshape((2*D*D,2)))\
                   .reshape((2*D,D,2))
        # update bL
        R = np.tensordot(np.conj(aRtil),np.tensordot(aRtil,NLR,(0,0)),
                         ((0,2),(2,1)))
        S = np.tensordot(np.conj(aRtil),np.tensordot(NLR,tmp,((0,2),(0,1))),
                         ((0,2),(0,2)))
        bLtil = spl.solve(R.transpose((0,3,2,1)).reshape((2*D*D,2*D*D)),
                          S.reshape((2*D*D,2))).reshape((2*D,D,2))
        cost = CTMRG.costfunction(aRtil,bLtil,NLR,tmp)
        if (abs(1-precost/cost))<10**(-10):  break
        precost = cost
    
    # reduced tensorを元に戻す
    a = np.tensordot(X,aRtil,(3,0)).transpose((0,3,2,1,4))
    b = np.tensordot(Y,bLtil,(3,0))
    #print('cost:',CTMRG.costfunction(aR,bL,NLR,tmp),'->',cost)
    return a, b

#### For Honeycomb Lattice Full Update #############
def FullUpdate_x(a,b,Ca,Ta1,Ta2,Cb,Tb1,Tb2,Cc,Tc1,Tc2,Cd,Td1,Td2,U,D):
    # create reduced tensors
    #      0           0
    #     /           /
    # 3--a--1  =  1--X-3---0-aR--1
    #   /|          /        |
    #  2 4         2         2
    tmp = a.transpose([0,3,2,1,4]);  sh = tmp.shape
    X, aR = spl.qr(tmp.reshape(np.prod(sh[:3]),sh[3]*sh[4]),
                   mode='economic')
    X = X.reshape(sh[:3]+(sh[3]*sh[4],))
    aR = aR.reshape((sh[3]*sh[4],)+sh[3:])
    #      0                    0
    #     /                    /
    # 3--b--1  =  1--bL-0---3-Y--1
    #   /|           |       /
    #  2 4           2      2
    sh = b.shape
    Y, bL = spl.qr(b.reshape(np.prod(sh[:3]),sh[3]*sh[4]),mode='economic')
    Y = Y.reshape(sh[:3]+(sh[3]*sh[4],))
    bL = bL.reshape((sh[3]*sh[4],)+sh[3:])
    
    # create an environment N_{LR}
    NLR = CTMRG.CreateNLR_x(Ca,Ta1,Ta2,Cb,Tb1,Tb2,Cc,Tc1,Tc2,Cd,Td1,Td2,X,Y,
                          CTMRG.ContractPhysBond(b,np.conj(b)),
                          CTMRG.ContractPhysBond(a,np.conj(a)))
    #NLR = GaugeFix(NLR)
    
    # update aR in the fixed bL
    tmp = np.tensordot(np.tensordot(aR,bL,(1,1)),U,((1,3),(0,1)))
    _, sv, bLtil = tensor_svd(tmp,(0,2),(1,3),D)
    # bLtilの初期値はUを作用させてSVDしたもの
    bLtil = (bLtil*np.sqrt(sv)[:,None,None]).transpose((1,0,2))
    precost = CTMRG.costfunction(aR,bL,NLR,tmp)
    while 1:
        R = np.tensordot(np.tensordot(NLR,bLtil,(2,0)),np.conj(bLtil),
                         ((2,4),(0,2)))
        S = np.tensordot(np.tensordot(NLR,tmp,((0,2),(0,1))),
                         np.conj(bLtil),((1,3),(0,2)))
        aRtil = spl.solve(R.transpose((1,3,0,2)).reshape((2*D*D,2*D*D)),
                          S.transpose((0,2,1)).reshape((2*D*D,2)))\
                   .reshape((2*D,D,2))
        # update bL
        R = np.tensordot(np.conj(aRtil),np.tensordot(aRtil,NLR,(0,0)),
                         ((0,2),(2,1)))
        S = np.tensordot(np.conj(aRtil),np.tensordot(NLR,tmp,((0,2),(0,1))),
                         ((0,2),(0,2)))
        bLtil = spl.solve(R.transpose((0,3,2,1)).reshape((2*D*D,2*D*D)),
                          S.reshape((2*D*D,2))).reshape((2*D,D,2))
        cost = CTMRG.costfunction(aRtil,bLtil,NLR,tmp)
        if (abs(1-precost/cost))<10**(-10):  break
        precost = cost
    
    # reduced tensorを元に戻す
    a = np.tensordot(X,aRtil,(3,0)).transpose((0,3,2,1,4))
    b = np.tensordot(Y,bLtil,(3,0))
    #print('cost:',CTMRG.costfunction(aR,bL,NLR,tmp),'->',cost)
    return a, b

def FullUpdate_y(a,b,Ca,Ta1,Ta2,Cb,Tb1,Tb2,Cc,Tc1,Tc2,Cd,Td1,Td2,U,D):
    # create reduced tensors
    #      0           0
    #     /           /
    # 3--a--1  =  1--X-3---0-aR--1
    #   /|          /        |
    #  2 4         2         2
    tmp = a.transpose([0,3,2,1,4]);  sh = tmp.shape
    X, aR = spl.qr(tmp.reshape(np.prod(sh[:3]),sh[3]*sh[4]),
                   mode='economic')
    X = X.reshape(sh[:3]+(sh[3]*sh[4],))
    aR = aR.reshape((sh[3]*sh[4],)+sh[3:])
    #      0                    0
    #     /                    /
    # 3--b--1  =  1--bL-0---3-Y--1
    #   /|           |       /
    #  2 4           2      2
    sh = b.shape
    Y, bL = spl.qr(b.reshape(np.prod(sh[:3]),sh[3]*sh[4]),mode='economic')
    Y = Y.reshape(sh[:3]+(sh[3]*sh[4],))
    bL = bL.reshape((sh[3]*sh[4],)+sh[3:])
    
    # create an environment N_{LR}
    NLR = CTMRG.CreateNLR_y(Ca,Ta1,Ta2,Cb,Tb1,Tb2,Cc,Tc1,Tc2,Cd,Td1,Td2,X,Y,
                          CTMRG.ContractPhysBond(b,np.conj(b)),
                          CTMRG.ContractPhysBond(a,np.conj(a)))
    #NLR = GaugeFix(NLR)
    
    # update aR in the fixed bL
    tmp = np.tensordot(np.tensordot(aR,bL,(1,1)),U,((1,3),(0,1)))
    _, sv, bLtil = tensor_svd(tmp,(0,2),(1,3),D)
    # bLtilの初期値はUを作用させてSVDしたもの
    bLtil = (bLtil*np.sqrt(sv)[:,None,None]).transpose((1,0,2))
    precost = CTMRG.costfunction(aR,bL,NLR,tmp)
    while 1:
        R = np.tensordot(np.tensordot(NLR,bLtil,(2,0)),np.conj(bLtil),
                         ((2,4),(0,2)))
        S = np.tensordot(np.tensordot(NLR,tmp,((0,2),(0,1))),
                         np.conj(bLtil),((1,3),(0,2)))
        aRtil = spl.solve(R.transpose((1,3,0,2)).reshape((2*D*D,2*D*D)),
                          S.transpose((0,2,1)).reshape((2*D*D,2)))\
                   .reshape((2*D,D,2))
        # update bL
        R = np.tensordot(np.conj(aRtil),np.tensordot(aRtil,NLR,(0,0)),
                         ((0,2),(2,1)))
        S = np.tensordot(np.conj(aRtil),np.tensordot(NLR,tmp,((0,2),(0,1))),
                         ((0,2),(0,2)))
        bLtil = spl.solve(R.transpose((0,3,2,1)).reshape((2*D*D,2*D*D)),
                          S.reshape((2*D*D,2))).reshape((2*D,D,2))
        cost = CTMRG.costfunction(aRtil,bLtil,NLR,tmp)
        if (abs(1-precost/cost))<10**(-10):  break
        precost = cost
    
    # reduced tensorを元に戻す
    a = np.tensordot(X,aRtil,(3,0)).transpose((0,3,2,1,4))
    b = np.tensordot(Y,bLtil,(3,0))
    #print('cost:',CTMRG.costfunction(aR,bL,NLR,tmp),'->',cost)
    return a, b

def FullUpdate_z(a,b,Ca,Ta1,Ta2,Cb,Tb1,Tb2,Cc,Tc1,Tc2,Cd,Td1,Td2,U,D):
    # create reduced tensors
    #      0           0
    #     /           /
    # 3--a--1  =  1--X-3---0-aR--1
    #   /|          /        |
    #  2 4         2         2
    tmp = a.transpose([0,3,2,1,4]);  sh = tmp.shape
    X, aR = spl.qr(tmp.reshape(np.prod(sh[:3]),sh[3]*sh[4]),
                   mode='economic')
    X = X.reshape(sh[:3]+(sh[3]*sh[4],))
    aR = aR.reshape((sh[3]*sh[4],)+sh[3:])
    #      0                    0
    #     /                    /
    # 3--b--1  =  1--bL-0---3-Y--1
    #   /|           |       /
    #  2 4           2      2
    sh = b.shape
    Y, bL = spl.qr(b.reshape(np.prod(sh[:3]),sh[3]*sh[4]),mode='economic')
    Y = Y.reshape(sh[:3]+(sh[3]*sh[4],))
    bL = bL.reshape((sh[3]*sh[4],)+sh[3:])
    
    # create an environment N_{LR}
    NLR = CTMRG.CreateNLR_z(Ca,Ta1,Ta2,Cb,Tb1,Tb2,Cc,Tc1,Tc2,Cd,Td1,Td2,X,Y,
                          CTMRG.ContractPhysBond(b,np.conj(b)),
                          CTMRG.ContractPhysBond(a,np.conj(a)))
    #NLR = GaugeFix(NLR)
    
    # update aR in the fixed bL
    tmp = np.tensordot(np.tensordot(aR,bL,(1,1)),U,((1,3),(0,1)))
    _, sv, bLtil = tensor_svd(tmp,(0,2),(1,3),D)
    # bLtilの初期値はUを作用させてSVDしたもの
    bLtil = (bLtil*np.sqrt(sv)[:,None,None]).transpose((1,0,2))
    precost = CTMRG.costfunction(aR,bL,NLR,tmp)
    while 1:
        R = np.tensordot(np.tensordot(NLR,bLtil,(2,0)),np.conj(bLtil),
                         ((2,4),(0,2)))
        S = np.tensordot(np.tensordot(NLR,tmp,((0,2),(0,1))),
                         np.conj(bLtil),((1,3),(0,2)))
        aRtil = spl.solve(R.transpose((1,3,0,2)).reshape((2*D*D,2*D*D)),
                          S.transpose((0,2,1)).reshape((2*D*D,2)))\
                   .reshape((2*D,D,2))
        # update bL
        R = np.tensordot(np.conj(aRtil),np.tensordot(aRtil,NLR,(0,0)),
                         ((0,2),(2,1)))
        S = np.tensordot(np.conj(aRtil),np.tensordot(NLR,tmp,((0,2),(0,1))),
                         ((0,2),(0,2)))
        bLtil = spl.solve(R.transpose((0,3,2,1)).reshape((2*D*D,2*D*D)),
                          S.reshape((2*D*D,2))).reshape((2*D,D,2))
        cost = CTMRG.costfunction(aRtil,bLtil,NLR,tmp)
        if (abs(1-precost/cost))<10**(-10):  break
        precost = cost
    
    # reduced tensorを元に戻す
    a = np.tensordot(X,aRtil,(3,0)).transpose((0,3,2,1,4))
    b = np.tensordot(Y,bLtil,(3,0))
    #print('cost:',CTMRG.costfunction(aR,bL,NLR,tmp),'->',cost)
    return a, b

#######################################
def GaugeFix(NLR):
    bdim = NLR.shape[0]
    NLRtil = (NLR + np.conj(NLR).transpose((2,3,0,1)))/2
    eiv, Pinv = spl.eigh(NLRtil.transpose((0,2,1,3))\
                         .reshape((bdim**2,bdim**2)))
    if eiv[0]<0:
        print('Gauge is to be fixed:',eiv)
        NLRtil = np.dot(Pinv*abs(~(eiv<0)*eiv)[None,:],np.conj(Pinv).T)\
                   .reshape(bdim,bdim,bdim,bdim).transpose((0,2,1,3))
    return NLRtil
