# -*- coding: utf-8 -*-
import scipy as sp
import scipy.linalg as spl
import numpy as np
import sys
import update as up
from msvd import *
from itertools import product
from scipy.linalg import expm
import json
from collections import OrderedDict
import torch
import itertools
"""
クラス Tensors_CTMに格納されているテンソルの定義


    C1--1     0--T1--2     0--C2
    |            |             |
    0            1             1

    2            0             0          
    |            |             |          
    T4--1     3--A--1     1 --T2       
    |            |             |        
    0            2             2          

    1           1              0
    |           |              | 
    C4--0    2--T3--0      1--C3

PEPSの順番
          0          0            0        0
         /          /             |        |
    3-- a --1   3-- b --1      3--A--1  3--B--1
      / |         / |             |        |
     2  4        2  4             2        2

Isometryの引数の定義

C1 -1       0--T21--2       0--T22--2       0- C2 
|               |1              |1              |
0                                               1

2                                               0
|                                               |
T12-1           A1            A2             1-T31
|                                               |
0                                               2

2                                               0
|                                               |    
T11-1           A4            A3            1-T32
|                                               |
0                                               2
 
1                                               0
|               |1          |1                  |
C4 -0       2--T42--0    2--T41--0           1- C3

  0    1  2
  |    |  |
  P    P_til
 | |    |
 1 2	0 
"""
class IPEPS():
    def __init__(self, sites, vertexToSite=None, lX=None, lY=None):
        r"""
        :param sites: map from elementary unit cell to on-site tensors
        :param vertexToSite: function mapping arbitrary vertex of a square lattice 
                             into a vertex within elementary unit cell
        :param lX: length of the elementary unit cell in X direction
        :param lY: length of the elementary unit cell in Y direction
        :param peps_args: ipeps configuration
        :param global_args: global configuration
        :type sites: dict[tuple(int,int) : torch.tensor]
        :type vertexToSite: function(tuple(int,int))->tuple(int,int)
        :type lX: int
        :type lY: int
        :type peps_args: PEPSARGS
        :type global_args: GLOBALARGS

        Member ``sites`` is a dictionary of non-equivalent on-site tensors
        indexed by tuple of coordinates (x,y) within the elementary unit cell.
        The index-position convetion for on-site tensors is defined as follows::

               u s 
               |/ 
            l--a--r  <=> a[s,u,l,d,r]
               |
               d
        
        where s denotes physical index, and u,l,d,r label four principal directions
        up, left, down, right in anti-clockwise order starting from up.
        Member ``vertexToSite`` is a mapping function from any vertex (x,y) on a square lattice
        passed in as tuple(int,int) to a corresponding vertex within elementary unit cell.
        
        On-site tensor of an IPEPS object ``wfc`` at vertex (x,y) is conveniently accessed 
        through the member function ``site``, which internally uses ``vertexToSite`` mapping::
            
            coord= (0,0)
            a_00= wfc.site(coord)

        By combining the appropriate ``vertexToSite`` mapping function with elementary unit 
        cell specified through ``sites``, various tilings of a square lattice can be achieved:: 
            
            # Example 1: 1-site translational iPEPS
            
            sites={(0,0): a}
            def vertexToSite(coord):
                return (0,0)
            wfc= IPEPS(sites,vertexToSite)
        
            # resulting tiling:
            # y\x -2 -1 0 1 2
            # -2   a  a a a a
            # -1   a  a a a a
            #  0   a  a a a a
            #  1   a  a a a a
            # Example 2: 2-site bipartite iPEPS
            
            sites={(0,0): a, (1,0): b}
            def vertexToSite(coord):
                x = (coord[0] + abs(coord[0]) * 2) % 2
                y = abs(coord[1])
                return ((x + y) % 2, 0)
            wfc= IPEPS(sites,vertexToSite)
        
            # resulting tiling:
            # y\x -2 -1 0 1 2
            # -2   A  b a b a
            # -1   B  a b a b
            #  0   A  b a b a
            #  1   B  a b a b
        
            # Example 3: iPEPS with 3x2 unit cell with PBC 
            
            sites={(0,0): a, (1,0): b, (2,0): c, (0,1): d, (1,1): e, (2,1): f}
            wfc= IPEPS(sites,lX=3,lY=2)
            
            # resulting tiling:
            # y\x -2 -1 0 1 2
            # -2   b  c a b c
            # -1   e  f d e f
            #  0   b  c a b c
            #  1   e  f d e f

        where in the last example a default setting for ``vertexToSite`` is used, which
        maps square lattice into elementary unit cell of size ``lX`` x ``lY`` assuming 
        periodic boundary conditions (PBC) along both X and Y directions.
        """
        self.sites= OrderedDict(sites)
        
        # TODO we infer the size of the cluster from the keys of sites. Is it OK?
        # infer the size of the cluster
        if lX is None or lY is None:
            min_x = min([coord[0] for coord in sites.keys()])
            max_x = max([coord[0] for coord in sites.keys()])
            min_y = min([coord[1] for coord in sites.keys()])
            max_y = max([coord[1] for coord in sites.keys()])
            self.lX = max_x-min_x + 1
            self.lY = max_y-min_y + 1
        else:
            self.lX = lX
            self.lY = lY

        if vertexToSite is not None:
            self.vertexToSite = vertexToSite
        else:
            def vertexToSite(coord):
                x = coord[0]
                y = coord[1]
                return ( (x + abs(x)*self.lX)%self.lX, (y + abs(y)*self.lY)%self.lY )
            self.vertexToSite = vertexToSite


###########################################################################    
class Tensors_CTM():

	def __init__(self, a):
		A = ContractPhysBond(a, a.conj())
		C1, C2, C3, C4, T1, T2, T3, T4 = initial_CTMRG(a)
		self.A  = A
		self.C1 = C1
		self.C2 = C2
		self.C3 = C3
		self.C4 = C4
		self.T1 = T1
		self.T2 = T2
		self.T3 = T3
		self.T4 = T4
		self.P  = T4
		self.P_til = T4

###########################################################################

def initial_iPEPS(D):
    ## random 複素数まで広げるとできないのはなぜ？
    a = np.random.random((D,D,D,D,2)) 
    b = np.random.random((D,D,D,D,2))

    ## ferro
    #a = np.array([1.,0.])[None,None,None,None,:];b=a
    ## flip
    #a = (np.ones(2)/np.sqrt(2))[None,None,None,None,:];b=a
    ## Anti-ferro
    #a = np.array([1.,0.])[None,None,None,None,:]
    #b = np.array([0.,1.])[None,None,None,None,:]    
    # vector lu, lr, ld, ll
    l = np.ones(a.shape[0], dtype=float)
    for i in np.arange(len(l)):    l[i] /= 10**i
    l /= np.sqrt(np.dot(l,l))
    
    return a,b,l,l,l,l

def initial_iPEPS_abcd(D):
    ## random
    #a = np.random.random((D,D,D,D,2))
    #b = np.random.random((D,D,D,D,2))
    ## ferro
    #a = np.array([1.,0.])[None,None,None,None,:];b=a
    ## flip
    #a = (np.ones(2)/np.sqrt(2))[None,None,None,None,:];b=a
    ## Anti-ferro
    a = np.array([1.,0.])[None,None,None,None,:]
    b = np.array([0.,1.])[None,None,None,None,:]    
    # vector lu, lr, ld, ll
    l = np.ones(a.shape[0], dtype=float)
    for i in np.arange(len(l)):    l[i] /= 10**i
    l /= np.sqrt(np.dot(l,l))
    
    return a

def initial_CTMRG(a):
    # tensor C's
    C1 = np.tensordot(a,np.conj(a),((0,3,4),(0,3,4)))\
           .transpose((1,3,0,2)).reshape((a.shape[2]**2,a.shape[1]**2))
    C2 = np.tensordot(a,np.conj(a),((0,1,4),(0,1,4)))\
           .transpose((1,3,0,2)).reshape((a.shape[3]**2,a.shape[2]**2))
    C3 = np.tensordot(a,np.conj(a),((1,2,4),(1,2,4)))\
           .transpose((0,2,1,3)).reshape((a.shape[0]**2,a.shape[3]**2))
    C4 = np.tensordot(a,np.conj(a),((2,3,4),(2,3,4)))\
           .transpose((1,3,0,2)).reshape((a.shape[1]**2,a.shape[0]**2))

    # tensor T's
    T1 = np.tensordot(a,np.conj(a),((3,4),(3,4))).transpose((2,5,1,4,0,3))\
            .reshape((a.shape[2]**2,a.shape[1]**2,a.shape[0]**2))

    T2 = np.tensordot(a,np.conj(a),((0,4),(0,4))).transpose((2,5,1,4,0,3))\
            .reshape((a.shape[3]**2,a.shape[2]**2,a.shape[1]**2))

    T3 = np.tensordot(a,np.conj(a),((1,4),(1,4))).transpose((0,3,2,5,1,4))\
            .reshape((a.shape[0]**2,a.shape[3]**2,a.shape[2]**2))

    T4 = np.tensordot(a,np.conj(a),((2,4),(2,4))).transpose((1,4,0,3,2,5))\
            .reshape((a.shape[1]**2,a.shape[0]**2,a.shape[3]**2))


    return C1, C2, C3, C4, T1, T2, T3, T4

def ContractPhysBond(x,conjy):
    b0 = x.shape[0]*conjy.shape[0]
    b1 = x.shape[1]*conjy.shape[1]
    b2 = x.shape[2]*conjy.shape[2]
    b3 = x.shape[3]*conjy.shape[3]
    xy = np.tensordot(x,conjy,(4,4))\
           .transpose((0,4,1,5,2,6,3,7)).reshape((b0,b1,b2,b3))
    return xy

###########################################################################

def Isometry(C1_n, C2_n, C3_n, C4_n):
	"""
	   3
	   |
	2--A--0  必ず'move'方向の反対側が'0'になるようにする!
	   |
	   1
	"""


	# upper halfとlower halfを作り、QR分解する
	upper_half = np.tensordot(C2_n, C1_n, ([0,1],[2,3]) )
	lower_half = np.tensordot(C3_n, C4_n, ([2,3],[0,1]) )
	#print( upper_half.shape, lower_half.shape )
	_,R_up = tensor_QR(upper_half, (0,1), (2,3), C1_n.shape[0]*C1_n.shape[1])
	_,R_low = tensor_QR(lower_half, (0,1), (2,3), C1_n.shape[0]*C1_n.shape[1])

	## Projection Operatorを作る
	U,s,Vdag =psvd(np.tensordot(R_up,R_low, ( [1,2],[1,2]) ) ,chi)
	#s = s/np.sqrt(np.dot(s,s))
	U = U*(np.sqrt(1./s))[None,:]
	Vdag = Vdag*(np.sqrt(1./s))[:,None]
	P = np.tensordot(U,R_up,([0],[0]) )
	P_til = np.tensordot(Vdag,R_low,([1],[0]) )
	#print( np.tensordot(P,P_til, [(1,2),(1,2)]),"\n")
	#print(s[0:4])

	return P, P_til

def CTM_corner(CTMs,x,y,A1,A2,A3,A4):

	x1=(x+1)%Lx ; x2=(x+2)%Lx ; x3=(x+3)%Lx
	y1=(y+1)%Ly ; y2=(y+2)%Ly ; y3=(y+3)%Ly


	## それぞれの四隅にまとめる
	
	Cc1 = np.transpose(
	    np.tensordot(
	        A1, np.tensordot(
	            CTMs[x1][y].T1, np.tensordot(
	                CTMs[x][y].C1, CTMs[x][y1].T4, ([0], [2])
	            ), ([0], [0])
	        ), ([2, 3], [3, 0])
	    ), [3, 1, 2, 0]
	)

	Cc2 = np.transpose(
	    np.tensordot(
	        A2, np.tensordot(
	            CTMs[x3][y1].T2, np.tensordot(
	                CTMs[x3][y].C2, CTMs[x2][y].T1, ([0], [2])
	            ), ([0], [0])
	        ), ([0, 3], [0, 3])
	    ), [3, 1, 2, 0]
	)

	Cc3 = np.transpose(
	    np.tensordot(
	        A3, np.tensordot(
	            CTMs[x2][y3].T3, np.tensordot(
	                CTMs[x3][y3].C3, CTMs[x3][y2].T2, ([0], [2])
	            ), ([0], [0])
	        ), ([0, 1], [3, 0])
	    ), [3, 1, 2, 0]
	)
	Cc4 = np.transpose(
	    np.tensordot(
	        A4, np.tensordot(
	            CTMs[x][y2].T4, np.tensordot(
	                CTMs[x][y3].C4, CTMs[x1][y3].T3, ([0], [2])
	            ), ([0], [0])
	        ), ([1, 2], [3, 0])
	    ), [3, 0, 2, 1]
	)

	return Cc1, Cc2, Cc3, Cc4

def CTM_update(C1,C4,T1,T4,T3,P,P_til,A):

	C1_new = np.transpose(
	    np.tensordot(
	        T1, np.tensordot(
	            C1, P_til, ([0], [1])
	        ), ([0, 1], [0, 2])
	    ), [1, 0]
	)
	C1_new = C1_new/np.amax( np.abs(C1_new) )

	C4_new = np.transpose(
	    np.tensordot(
	        T3, np.tensordot(
	            C4, P, ([1], [1])
	        ), ([1, 2], [2, 0])
	    ), [0, 1]
	)
	C4_new = C4_new/np.amax( np.abs(C4_new) )

	T4_new = np.transpose(
	    np.tensordot(
	        P, np.tensordot(
	            A, np.tensordot(
	                T4, P_til, ([0], [1])
	            ), ([1, 2], [3, 0])
	        ), ([1, 2], [2, 1])
	    ), [2, 1, 0]
	)
	T4_new = T4_new/np.amax( np.abs(T4_new) )

	return C1_new, C4_new, T4_new

def CTMRG(CTMs, Lx, Ly):
	######## Left Move ###########
	for x in range(Lx):
		x1 = (x+1)%Lx ; x2 = (x+2)%Lx
		for y in range(Ly):
			y1 = (y+1)%Ly ; y2 = (y+2)%Ly
			C1_n, C2_n, C3_n, C4_n = CTM_corner(CTMs,x,y,\
				CTMs[x1][y1].A.transpose(1,2,3,0) ,\
				CTMs[x2][y1].A.transpose(1,2,3,0) ,\
				CTMs[x2][y2].A.transpose(1,2,3,0) ,\
				CTMs[x1][y2].A.transpose(1,2,3,0))
			CTMs[x][y1].P , CTMs[x][y1].P_til = Isometry(C1_n, C2_n, C3_n, C4_n)
			#print( "left ", CTMs[x][y1].P.shape, CTMs[x][y1].P_til.shape, (x,y) )

		for y in range(Ly):
			x1 = (x+1)%Lx ; y1 = (y-1)%Ly

			CTMs[x1][y].C1, CTMs[x1][y].C4, CTMs[x1][y].T4 \
			= CTM_update(CTMs[x][y].C1, \
				CTMs[x][y].C4, CTMs[x1][y].T1, CTMs[x][y].T4,\
				CTMs[x1][y].T3, CTMs[x][y1].P, CTMs[x][y].P_til, CTMs[x1][y].A.transpose(1,2,3,0))



	
	######## Up Move #############
	for y in range(Ly):
		y1 = (y+1)%Ly ; y2 = (y+2)%Ly
		for x in range(Lx):
			x1 = (x+1)%Lx ; x2 = (x+2)%Lx
			C1_n, C2_n, C3_n, C4_n = CTM_corner(CTMs,x,y,\
				CTMs[x2][y1].A.transpose(2,3,0,1) ,\
				CTMs[x2][y2].A.transpose(2,3,0,1) ,\
				CTMs[x1][y2].A.transpose(2,3,0,1) ,\
				CTMs[x1][y1].A.transpose(2,3,0,1))
			CTMs[x2][y].P , CTMs[x2][y].P_til = Isometry(C2_n, C3_n, C4_n, C1_n)
			#print( "UP ", CTMs[x2][y].P.shape, CTMs[x2][y].P_til.shape, (x,y) )
	
		
		for x in range(Lx):
			x3 = (x+3)%Lx ; x4= (x+4)%Lx
			y1 = (y+1)%Ly 
			CTMs[x3][y1].C2, CTMs[x3][y1].C1, CTMs[x3][y1].T1 \
			= CTM_update(CTMs[x3][y].C2, \
				CTMs[x3][y].C1, CTMs[x3][y1].T2, CTMs[x3][y].T1,\
				CTMs[x3][y1].T4, CTMs[x4][y].P, CTMs[x3][y].P_til, CTMs[x3][y1].A.transpose(2,3,0,1))


		
	######## Right Move ##########
	for x in range(Lx):
		x3 = (x+3)%Lx
		for y in range(Ly):
			y2 = (y+2)%Ly
			C1_n, C2_n, C3_n, C4_n = CTM_corner(CTMs,x,y,\
				CTMs[x2][y2].A.transpose(3,0,1,2) ,\
				CTMs[x1][y2].A.transpose(3,0,1,2) ,\
				CTMs[x1][y1].A.transpose(3,0,1,2) ,\
				CTMs[x2][y1].A.transpose(3,0,1,2))
			CTMs[x3][y2].P , CTMs[x3][y2].P_til = Isometry(C3_n, C4_n, C1_n, C2_n)
			#print( "Right ", CTMs[x3][y2].P.shape, CTMs[x3][y2].P_til.shape, (x,y) )


		for y in range(Ly):
			x2 = (x+2)%Lx ; x3 = (x+3)%Lx 
			y3 = (y+3)%Ly ; y4 = (y+4)%Ly

			#print(CTMs[x3][y3].C3.shape, CTMs[x2][y3].P_til.shape )
			CTMs[x2][y3].C3, CTMs[x2][y3].C2, CTMs[x2][y3].T2 \
			= CTM_update(CTMs[x3][y3].C3, \
				CTMs[x3][y3].C2, CTMs[x2][y3].T3, CTMs[x3][y3].T2,\
				CTMs[x2][y3].T1, CTMs[x3][y4].P, CTMs[x3][y3].P_til, CTMs[x2][y3].A.transpose(3,0,1,2))

	######## Down Move ###########
	for y in range(Ly):
		y3 = (y+3)%Ly ; y2 = (y+2)%Ly ; y1 = (y+1)%Ly
		for x in range(Lx):
			x1 = (x+1)%Lx ; x2 = (x+2)%Lx
			C1_n, C2_n, C3_n, C4_n = CTM_corner(CTMs,x,y,\
				CTMs[x1][y2].A ,\
				CTMs[x1][y1].A ,\
				CTMs[x2][y1].A ,\
				CTMs[x2][y2].A)
			CTMs[x1][y3].P , CTMs[x1][y3].P_til = Isometry(C4_n, C1_n, C2_n, C3_n)
			#print( "Down ", CTMs[x1][y3].P.shape, CTMs[x1][y3].P_til.shape, (x,y) )	

		for x in range(Lx):
			x1 = (x-1)%Lx 
	
			CTMs[x][y2].C4, CTMs[x][y2].C3, CTMs[x][y2].T3 \
			= CTM_update(CTMs[x][y3].C4, \
				CTMs[x][y3].C3, CTMs[x][y2].T4, CTMs[x][y3].T3,\
				CTMs[x][y2].T2, CTMs[x1][y3].P, CTMs[x][y3].P_til, CTMs[x][y2].A)


	######## Expectation Value ###########

def ComputeQuantities(a,b,c,d,A,B,C,D,C1,C2,C3,C4,T11,T12,T21,T22,T31,T32,T41,T42,H):

   
    # environment is contracted into four tensors
    Cc1 = np.tensordot(np.tensordot(T12,C1,(2,0)),T21,(2,0))
    Cc2 = np.tensordot(np.tensordot(T22,C2,(2,0)),T31,(2,0))
    Cc3 = np.tensordot(np.tensordot(T32,C3,(2,0)),T41,(2,0))
    Cc4 = np.tensordot(np.tensordot(T42,C4,(2,0)),T11,(2,0))

    sx,sy,sz=spin_operators(0.5)
    # <Ψ|Ψ>
    psipsi = ContractNetwork(A,B,C,D,Cc1,Cc2,Cc3,Cc4)
    
    # <Ψ|sx|Ψ>
    #sx = np.array([[0.,1.],[1.,0.]])
    mx, mxb = sandwitch(a,b,c,d,sx,A,B,C,D,Cc1,Cc2,Cc3,Cc4)

    # <Ψ|sy|Ψ>
    #sy = np.array([[0.,-1.0j],[1.0j,0.]])
    my,myb = sandwitch(a,b,c,d,sy,A,B,C,D,Cc1,Cc2,Cc3,Cc4)
    
    # <Ψ|sz|Ψ>
    #sz = np.diag([1.,-1.])
    mz,mzb = sandwitch(a,b,c,d,sz,A,B,C,D,Cc1,Cc2,Cc3,Cc4)

    #print mxa/psipsi, mxc/psipsi, mya/psipsi, myc/psipsi, mza/psipsi, mzc/psipsi
    
    # <Ψ|H|Ψ>
    c1 = np.tensordot(Cc1,A,((1,2),(3,0))).transpose([0,3,2,1])
    c2 = np.tensordot(Cc2,B,((1,2),(0,1))).transpose([0,3,2,1])
    c3 = np.tensordot(Cc3,C,((1,2),(1,2))).transpose([0,2,3,1])
    c4 = np.tensordot(Cc4,D,((1,2),(2,3))).transpose([0,3,2,1])
    def newshape(C):
        D1 = int(np.sqrt(C.shape[1]));  D2 = int(np.sqrt(C.shape[2]))
        return C.reshape((C.shape[0],D1,D1,D2,D2,C.shape[3]))
    d1 = np.tensordot(a,np.tensordot(newshape(Cc1),np.conj(a),([2,4],[3,0])),
                      ([0,3],[2,1])).transpose([3,1,6,0,5,4,2,7])
    d2 = np.tensordot(b,np.tensordot(newshape(Cc2),np.conj(b),([2,4],[0,1])),
                      ([0,1],[1,2])).transpose([3,1,6,0,5,4,2,7])
    d3 = np.tensordot(c,np.tensordot(newshape(Cc3),np.conj(c),([2,4],[1,2])),
                      ([1,2],[1,2])).transpose([3,0,5,1,6,4,2,7])
    d4 = np.tensordot(d,np.tensordot(newshape(Cc4),np.conj(d),([2,4],[2,3])),
                      ([2,3],[1,2])).transpose([3,1,6,0,5,4,2,7])
    def ContractWithH(da,db,cc,cd,H):

        tmp = np.tensordot(H,np.tensordot(da,np.tensordot(db,\
                    np.tensordot(newshape(cc),newshape(cd),([3,4,5],[1,2,0])),
                                                          ([3,4,5],[1,2,0])),
                                          ([0,1,2,3,4,5],[7,5,6,1,2,0])),
                           ([0,1,2,3],[0,2,1,3]))
        return tmp
    psiHpsi = ContractWithH(d1,d2,c3,c4,H)+ContractWithH(d2,d3,c4,c1,H)+\
              ContractWithH(d3,d4,c1,c2,H)+ContractWithH(d4,d1,c2,c3,H)
    #print(ContractWithH(d1,d2,c3,c4,H)/psipsi, ContractWithH(d2,d3,c4,c1,H)/psipsi, ContractWithH(d3,d4,c1,c2,H)/psipsi, ContractWithH(d4,d1,c2,c3,H)/psipsi)
   
    return np.real(psipsi), np.real(mx/psipsi), np.real(my/psipsi), np.real(mz/psipsi), np.real(psiHpsi/psipsi/4*2)

def sandwitch(a,b,c,d,Op,A,B,C,D,Cc1,Cc2,Cc3,Cc4):
    # impurity tensor
    impa = ContractPhysBond(np.tensordot(a,Op,(4,0)),np.conj(a))
    impb = ContractPhysBond(np.tensordot(b,Op,(4,0)),np.conj(b))
    impc = ContractPhysBond(np.tensordot(c,Op,(4,0)),np.conj(c))
    impd = ContractPhysBond(np.tensordot(d,Op,(4,0)),np.conj(d))

    if model=="I":
    # sandwitch and average
        psi_O_psi = ContractNetwork(impa,B,C,D,Cc1,Cc2,Cc3,Cc4) \
                  + ContractNetwork(A,impb,C,D,Cc1,Cc2,Cc3,Cc4) \
                  + ContractNetwork(A,B,impc,D,Cc1,Cc2,Cc3,Cc4) \
                  + ContractNetwork(A,B,C,impd,Cc1,Cc2,Cc3,Cc4)
        return psi_O_psi/4, psi_O_psi/4

    if model=="H":
    # sandwitch and average
        if lattice=="squ":
	        psi_Oa_psi = ContractNetwork(impa,B,C,D,Cc1,Cc2,Cc3,Cc4) \
	                   + ContractNetwork(A,B,impc,D,Cc1,Cc2,Cc3,Cc4)
	        psi_Ob_psi = ContractNetwork(A,impb,C,D,Cc1,Cc2,Cc3,Cc4) \
	        		   + ContractNetwork(A,B,C,impd,Cc1,Cc2,Cc3,Cc4)
	        return psi_Oa_psi/2, psi_Ob_psi/2
        if lattice=="tri": 
	        psi_Oa_psi = ContractNetwork(impa,B,C,D,Cc1,Cc2,Cc3,Cc4) 
	        psi_Ob_psi = ContractNetwork(A,impb,C,D,Cc1,Cc2,Cc3,Cc4) 
	        return psi_Oa_psi, psi_Ob_psi
    #print('{0:4f},{1:4f} {2:4f}, {3:4f}'.format(ContractNetwork(impa,B,C,D,Cc1,Cc2,Cc3,Cc4) , ContractNetwork(A,B,impc,D,Cc1,Cc2,Cc3,Cc4), \
    #ContractNetwork(A,impb,C,D,Cc1,Cc2,Cc3,Cc4), ContractNetwork(A,B,C,impd,Cc1,Cc2,Cc3,Cc4)) )

def ContractNetwork(P,Q,R,S,Cc1,Cc2,Cc3,Cc4):
    # C1C1C1--C2C2C2
    # C1  |   |   C2
    # C1--P---Q---C2
    # |   |   |   |
    # C4--S---R---C3
    # C4  |   |   C3
    # C4C4C4--C3C3C3
    c1 = np.tensordot(Cc1,P,((1,2),(3,0)))
    c2 = np.tensordot(Cc2,Q,((1,2),(0,1)))
    c3 = np.tensordot(Cc3,R,((1,2),(1,2)))
    c4 = np.tensordot(Cc4,S,((1,2),(2,3)))
    
    c4 = np.tensordot(c4,c3,((0,3),(1,3)))
    c1 = np.tensordot(c1,c2,((1,2),(0,3)))
    return np.einsum('ijkl,ijkl',c1,c4)

def local_quantities(C1,C2,C3,C4,T1,T2,T3,T4,a,Op):

	Aimp = ContractPhysBond(np.tensordot(a,Op,(4,0)),np.conj(a))
	A    = ContractPhysBond(a, a.conj())

	local = np.tensordot( 
	    Aimp, np.tensordot(
	        np.tensordot(
	            T1, np.tensordot(
	                C1, T4, ([0], [2])
	            ), ([0], [0])
	        ), np.tensordot(
	            np.tensordot(
	                C2, T2, ([1], [0])
	            ), np.tensordot(
	                C3, np.tensordot(
	                    T3, C4, ([2], [0])
	                ), ([1], [0])
	            ), ([2], [0])
	        ), ([1, 2], [0, 3])
	    ), ([0, 1, 2, 3], [0, 2, 3, 1])
	)

	psi_psi = np.tensordot( 
	    A, np.tensordot(
	        np.tensordot(
	            T1, np.tensordot(
	                C1, T4, ([0], [2])
	            ), ([0], [0])
	        ), np.tensordot(
	            np.tensordot(
	                C2, T2, ([1], [0])
	            ), np.tensordot(
	                C3, np.tensordot(
	                    T3, C4, ([2], [0])
	                ), ([1], [0])
	            ), ([2], [0])
	        ), ([1, 2], [0, 3])
	    ), ([0, 1, 2, 3], [0, 2, 3, 1])
	)

	return np.real(local/psi_psi)

###########################################################################
# 横磁場イジング模型
def spin_operators(S):
	"""" Returns the spin operators  """	
	d = int(np.rint(2*S + 1))
	dz = np.zeros(d);  mp = np.zeros(d-1)

	for n in range(d-1):
		dz[n] = S - n
		mp[n] = np.sqrt((2.0*S - n)*(n + 1.0))

		dz[d - 1] = - S
	Sp = np.diag(mp,1);   Sm = np.diag(mp,-1)
	Sx = 0.5*(Sp + Sm);   Sy = -0.5j*(Sp - Sm)
	Sz = np.diag(dz)    

	return Sx, Sy, Sz

def ImagTimeEvol(hx,dt):
    tmp = np.sqrt(4+hx**2)
    c = np.cosh(dt);  s = np.sinh(dt)
    x = np.cosh(dt*tmp/2);  y = np.sinh(dt*tmp/2)/tmp
    # 虚時間発展演算子をつくる
    U = np.zeros((2,2,2,2), dtype=float)
    U[(0,0,0,0)] = U[(1,1,1,1)] = (c+s)/2+y
    U[(0,0,1,1)] = U[(1,1,0,0)] = (-c-s)/2+y
    U[(0,1,0,1)] = U[(1,0,1,0)] = (c-s)/2-y
    U[(0,1,1,0)] = U[(1,0,0,1)] = (-c+s)/2-y
    for idx,_ in np.ndenumerate(U):
        if sum(idx)%2==0:    U[idx] += x/2
        else:    U[idx] += hx * y / 2
    return U

def Hamiltonian_Ising(h):
    H = np.zeros((2,2,2,2), dtype=float)
    H[(0,0,0,0)] = H[(1,1,1,1)] = -1
    H[(0,1,0,1)] = H[(1,0,1,0)] = 1
    for idx,_ in np.ndenumerate(H):
        if sum(idx)%2!=0:    H[idx] = -h/4
    return H

def Ham_Ising(h,dt):
    sx,sy,sz=spin_operators(0.5)
    I=np.eye(2)
    H = -np.kron(sz,sz)-0.25*h*(np.kron(sx,I)+np.kron(sx,I)  )
    U=expm(-dt*H).reshape(2,2,2,2)

    return H.reshape(2,2,2,2), U

###########################################################################
# ハイゼンベルグ模型
def Hamiltonian_Hiese(dt,J):

    Sx, Sy, Sz = spin_operators(0.5)
    H =J*np.kron(Sx,Sx) + J*np.kron(Sy,Sy) + J*np.kron(Sz,Sz)  
    U=expm(-dt*H).reshape(2,2,2,2)

    return H.reshape(2,2,2,2), U

# 三角格子用
def Hamiltonian_Hiese_nn(dt,J):

    Sx, Sy, Sz = spin_operators(0.5)
    I = np.eye(2,2)
    H = J*np.kron(np.kron(Sx,I), Sx) + J*np.kron(np.kron(Sy,I), Sy) + J*np.kron(np.kron(Sz,I), Sz)  
    U = expm(-dt*H).reshape(2,2,2,2,2,2)
    u,s,v = tensor_svd(U, (0,3,4), (1,2,5), 8)
    s = np.sqrt(s)
    u = u*s[None,None,None,:]
    v = v*s[:,None,None,None]
    u = u.transpose(0,3,1,2)
    v = v.transpose(1,2,0,3)

    return u,v
###########################################################################
# Potts模型
def potts_TN(D,beta,J):

    Ta = np.zeros((D,D,D,D),dtype=float) # Ta_urdl
    T_imp = np.zeros((D,D,D,D),dtype=float) # Ta_urdl
    T_imp_dag = np.zeros((D,D,D,D),dtype=float) # Ta_urdl
    TT = np.ones((D,D,D,D),dtype=float)


    Lam_0 = D + np.exp(beta) -1.0   ; Lam_0 = np.sqrt(Lam_0)
    Lam_1 = np.exp(beta) -1.0       ; Lam_1 = np.sqrt(Lam_1)
    lam = Lam_1 * np.ones(D)        ; lam[0] = Lam_0

    for idx,_ in np.ndenumerate(Ta):
        if( (idx[0]+idx[1])%D == (idx[2]+idx[3])%D):
            Ta[idx] = lam[idx[0]]*lam[idx[1]]*lam[idx[2]]*lam[idx[3]]/float(D)
        if( (idx[0]+idx[1])%D == (idx[2]+idx[3]+1)%D):
            T_imp[idx] = lam[idx[0]]*lam[idx[1]]*lam[idx[2]]*lam[idx[3]]/float(D)
        if( (idx[0]+idx[1])%D == (idx[2]+idx[3]-1)%D):
            T_imp_dag[idx] = lam[idx[0]]*lam[idx[1]]*lam[idx[2]]*lam[idx[3]]/float(D)

        TT[idx] = lam[idx[0]]*lam[idx[1]]*lam[idx[2]]*lam[idx[3]]/float(D)


    return Ta, T_imp, T_imp_dag
	#return Ta, TT, TTT
def write_ipeps(state, outputfile, aux_seq=[0,1,2,3], tol=1.0e-14, normalize=False):
    r"""
    :param state: wavefunction to write out in json format
    :param outputfile: target file
    :param aux_seq: array specifying order in which the auxiliary indices of on-site tensors 
                    will be stored in the `outputfile`
    :param tol: minimum magnitude of tensor elements which are written out
    :param normalize: if True, on-site tensors are normalized before writing
    :type state: IPEPS
    :type ouputfile: str or Path object
    :type aux_seq: list[int]
    :type tol: float
    :type normalize: bool

    Parameter ``aux_seq`` defines the order of auxiliary indices relative to the convention 
    fixed in tn-torch in which the tensor elements are written out::
    
         0
        1A3 <=> [up, left, down, right]: aux_seq=[0,1,2,3]
         2
        
        for alternative order, eg.
        
         1
        0A2 <=> [left, up, right, down]: aux_seq=[1,0,3,2] 
         3

    """
    asq = [x+1 for x in aux_seq]
    json_state=dict({"lX": state.lX, "lY": state.lY, "sites": []})
    
    site_ids=[]
    site_map=[]
    for nid,coord,site in [(t[0], *t[1]) for t in enumerate(state.sites.items())]:
        if normalize:
            site= site/torch.max(torch.abs(site))

        json_tensor=dict()
        
        tdims = site.size()
        tlength = tdims[0]*tdims[1]*tdims[2]*tdims[3]*tdims[4]
        
        site_ids.append(f"A{nid}")
        site_map.append(dict({"siteId": site_ids[-1], "x": coord[0], "y": coord[1]} ))
        json_tensor["siteId"]=site_ids[-1]
        json_tensor["physDim"]= tdims[0]
        # assuming all auxBondDim are identical
        json_tensor["auxDim"]= tdims[1]
        json_tensor["numEntries"]= tlength
        entries = []
        elem_inds = list(itertools.product( *(range(i) for i in tdims) ))
        for ei in elem_inds:
            entries.append(f"{ei[0]} {ei[asq[0]]} {ei[asq[1]]} {ei[asq[2]]} {ei[asq[3]]}"\
                +f" {site[ei[0]][ei[1]][ei[2]][ei[3]][ei[4]]}")
            
        json_tensor["entries"]=entries
        json_state["sites"].append(json_tensor)

    json_state["siteIds"]=site_ids
    json_state["map"]=site_map

    with open(outputfile,'w') as f:
        json.dump(json_state, f, indent=4, separators=(',', ': '))
##  main function
if __name__=="__main__":

	if len(sys.argv) < 2:  D = 2
	else:  D = int(sys.argv[1])
	if len(sys.argv) < 3:  Hx = 0.0
	else:  Hx = float(sys.argv[2])
	if len(sys.argv) < 4:  dt = 0.01
	else:  dt = float(sys.argv[3])
	if len(sys.argv) < 5:  chi = D*10
	else:  chi = int(sys.argv[4])
	if len(sys.argv) < 6:  maxstepTEBD = 4000#10**4
	else:  maxstepTEBD = int(sys.argv[5])
	if len(sys.argv) < 7:  maxstepCTM = 100
	else:  maxstepCTM = int(sys.argv[6])
	if len(sys.argv) < 8:  model="H"
	else:  model = (sys.argv[7])
	if len(sys.argv) < 9:  lattice="tri"
	else:  lattice = (sys.argv[8])

	Lx   = 3
	Ly   = 3
	eps_TEBD = 10**(-10);  eps_CTM = 10**(-10)
	CTMs = [[0 for y in range(Ly)]  for x in range(Lx)  ]
	SU_ten = [[0 for y in range(Ly)]  for x in range(Lx)  ]

	#####  ハミルトニアン ####
	## イジング
	if model =="I":
		U = ImagTimeEvol(Hx,dt)
		Ham,U = Ham_Ising(Hx,dt)

	## ハイゼンベルグ
	if model == "H":
		Ham,U2 = Hamiltonian_Hiese(dt,J=1.0)
		Ham,U = Hamiltonian_Hiese(dt,J=1.0)
		U1,U2 = Hamiltonian_Hiese_nn(dt,J=1.0)

	"""
	SU_ten =[ ["a","c","b"], ["c","b","a"],["b","a","c"] ]
	for i in range(Lx):
		for j in range(Ly):
			print( (i,j), SU_ten[i][j])
	exit()
	"""
	

	########## Simple Update ############################

	a, b, l1, l2, l3, l4 = initial_iPEPS(D)
	c, d, l5, l6, l7, l8 = initial_iPEPS(D)

	pre_lu,pre_lr,pre_ld,pre_ll = l1,l2,l3,l4

	## 正方格子とチェックボート格子両方に対応できる(チェックボートは4×4の単位胞が必要)
	# 初期状態をNeel状態にする
	"""
	for i in range(maxstepTEBD):
	    #a,b,lu,lr,ld,ll = up.SimpleUpdate(a,b,lu,lr,ld,ll,U,D)
	    a,b,c,d,l1,l2,l3,l4,l5,l6,l7,l8 = up.SimpleUpdate_abcd(a,b,c,d,l1,l2,l3,l4,l5,l6,l7,l8,U,D)

	    print(i,l1,"\n",l2,"\n",l3,"\n",l4)
	    try:
	        res1,res2,res3,res4 = sum(abs(l1-pre_lu)),sum(abs(l2-pre_lr)),\
	                              sum(abs(l3-pre_ld)),sum(abs(l4-pre_ll))
	    except ValueError:
	        res1=res2=res3=res4 = float('inf')
	    # decide whether converge or not
	    if res1<eps_TEBD and res2<eps_TEBD \
	       and res3<eps_TEBD and res4<eps_TEBD:
	        break
	    pre_lu,pre_lr,pre_ld,pre_ll = l1,l2,l3,l4
	"""
	## 正方格子の場合
	if lattice == "squ":
		for i in range(maxstepTEBD):

		    #a,b,lu,lr,ld,ll = up.SimpleUpdate(a,b,lu,lr,ld,ll,U,D)
		    a,b,c,d,l1,l2,l3,l4,l5,l6,l7,l8 = up.SimpleUpdate_abcd(a,b,c,d,l1,l2,l3,l4,l5,l6,l7,l8,U,D)

		    #a,b,c,d,l1,l2,l3,l4,l5,l6,l7,l8 = up.SimpleUpdate_abcd_NN(a,b,c,d,l1,l2,l3,l4,l5,l6,l7,l8,U2,D)
		    #b,a,d,c,l7,l4,l5,l2,l3,l8,l1,l6 = up.SimpleUpdate_abcd_NN(b,a,d,c,l7,l4,l5,l2,l3,l8,l1,l6,U2,D)
		    #d,c,b,a,l5,l6,l7,l8,l1,l2,l3,l4 = up.SimpleUpdate_abcd_NN(d,c,b,a,l5,l6,l7,l8,l1,l2,l3,l4,U2,D)
		    #c,d,a,b,l3,l8,l1,l6,l7,l4,l5,l2 = up.SimpleUpdate_abcd_NN(c,d,a,b,l3,l8,l1,l6,l7,l4,l5,l2,U2,D)
		    print(i,l1,"\n",l2,"\n",l3,"\n",l4,"\n",l5,"\n",l6,"\n",l7,"\n",l8,"\n")
		    try:
		        res1,res2,res3,res4 = sum(abs(l1-pre_lu)),sum(abs(l2-pre_lr)),\
		                              sum(abs(l3-pre_ld)),sum(abs(l4-pre_ll))
		    except ValueError:
		        res1=res2=res3=res4 = float('inf')
		    # decide whether converge or not
		    if res1<eps_TEBD and res2<eps_TEBD \
		       and res3<eps_TEBD and res4<eps_TEBD:
		        break
		    pre_lu,pre_lr,pre_ld,pre_ll = l1,l2,l3,l4

		# 収束しているなら、４つの特異値ベクトルはほとんど同じ
		ave_l = (l1+l2+l3+l4)/4
		if sum(abs(ave_l-l1))/len(l1)<10**(-5) and \
		   sum(abs(ave_l-l2))/len(l2)<10**(-5) and \
		   sum(abs(ave_l-l3))/len(l3)<10**(-5) and \
		   sum(abs(ave_l-l4))/len(l4)<10**(-5) :
		    print('-- iTEBD has converged --\n')
		else:  print('-- iTEBD has NOT converged --\n')

		############################################
		l1 = np.sqrt(l1);  l2 = np.sqrt(l2);   l3 = np.sqrt(l3);  l4 = np.sqrt(l4)
		l5 = np.sqrt(l5);  l6 = np.sqrt(l6);   l7 = np.sqrt(l7);  l8 = np.sqrt(l8)
		
		a *= l1[:,None,None,None,None]*l2[None,:,None,None,None] \
		     *l3[None,None,:,None,None]*l4[None,None,None,:,None]
		b *= l7[:,None,None,None,None]*l4[None,:,None,None,None] \
		     *l5[None,None,:,None,None]*l2[None,None,None,:,None]
		c *= l3[:,None,None,None,None]*l8[None,:,None,None,None] \
		     *l1[None,None,:,None,None]*l6[None,None,None,:,None]
		d *= l5[:,None,None,None,None]*l6[None,:,None,None,None] \
		     *l7[None,None,:,None,None]*l8[None,None,None,:,None]

	## 三角格子の場合
	if lattice == "tri":

		for i in range(maxstepTEBD):

		    a,c,b,l3,l5 = up.SimpleUpdate_triangular(a,c,b,l1,l2,l3,l4,l5,l6,U,U1,U2,D)
		    c,b,a,l6,l4 = up.SimpleUpdate_triangular(c,b,a,l3,l5,l6,l2,l4,l1,U,U1,U2,D)
		    b,a,c,l1,l2 = up.SimpleUpdate_triangular(b,a,c,l6,l4,l1,l5,l2,l3,U,U1,U2,D)
		    
		    #a,c,b,l1,l2,l3,l4,l5,l6 = up.SimpleUpdate_abcd_triangular_right_upper(a,c,b,l1,l2,l3,l4,l5,l6,U,D)
		    #c,b,a,l3,l5,l6,l2,l4,l1 = up.SimpleUpdate_abcd_triangular_right_upper(c,b,a,l3,l5,l6,l2,l4,l1,U,D)
		    #b,a,c,l6,l4,l1,l5,l2,l3 = up.SimpleUpdate_abcd_triangular_right_upper(b,a,c,l6,l4,l1,l5,l2,l3,U,D)

		    #a,c,b,l1,l2,l3,l4,l5,l6 = up.SimpleUpdate_abcd_triangular_left_lowwer(a,c,b,l1,l2,l3,l4,l5,l6,U,D)
		    #c,b,a,l3,l5,l6,l2,l4,l1 = up.SimpleUpdate_abcd_triangular_left_lowwer(c,b,a,l3,l5,l6,l2,l4,l1,U,D)
		    #b,a,c,l6,l4,l1,l5,l2,l3 = up.SimpleUpdate_abcd_triangular_left_lowwer(b,a,c,l6,l4,l1,l5,l2,l3,U,D)

		    print(i,l1,"\n",l2,"\n",l3,"\n",l4,"\n",l5,"\n",l6,"\n")
		    try:
		        res1,res2,res3,res4 = sum(abs(l1-pre_lu)),sum(abs(l2-pre_lr)),\
		                              sum(abs(l3-pre_ld)),sum(abs(l4-pre_ll))
		    except ValueError:
		        res1=res2=res3=res4 = float('inf')
		    # decide whether converge or not
		    if res1<eps_TEBD and res2<eps_TEBD \
		       and res3<eps_TEBD and res4<eps_TEBD:
		        break
		    pre_lu,pre_lr,pre_ld,pre_ll = l1,l2,l3,l4

		# 収束しているなら、４つの特異値ベクトルはほとんど同じ
		ave_l = (l1+l2+l3+l4)/4
		if sum(abs(ave_l-l1))/len(l1)<10**(-5) and \
		   sum(abs(ave_l-l2))/len(l2)<10**(-5) and \
		   sum(abs(ave_l-l3))/len(l3)<10**(-5) and \
		   sum(abs(ave_l-l4))/len(l4)<10**(-5) :
		    print('-- iTEBD has converged --\n')
		else:  print('-- iTEBD has NOT converged --\n')

		############################################
		l1 = np.sqrt(l1);  l2 = np.sqrt(l2);   l3 = np.sqrt(l3);  l4 = np.sqrt(l4)
		l5 = np.sqrt(l5);  l6 = np.sqrt(l6);   l7 = np.sqrt(l7);  l8 = np.sqrt(l8)

		## 三角格子の場合
		a *= l1[:,None,None,None,None]*l2[None,:,None,None,None] \
		     *l3[None,None,:,None,None]*l4[None,None,None,:,None]
		b *= l6[:,None,None,None,None]*l4[None,:,None,None,None] \
		     *l1[None,None,:,None,None]*l5[None,None,None,:,None]
		c *= l3[:,None,None,None,None]*l5[None,:,None,None,None] \
		     *l6[None,None,:,None,None]*l2[None,None,None,:,None]


	a=np.real(a.transpose(4,0,3,2,1))
	b=np.real(b.transpose(4,0,3,2,1))
	c=np.real(c.transpose(4,0,3,2,1))
	a = torch.from_numpy(a.astype(np.float32)).clone()
	b = torch.from_numpy(b.astype(np.float32)).clone()
	c = torch.from_numpy(c.astype(np.float32)).clone()
	sites = {(0,0): a, (1,0): b, (2,0): c}
	state = IPEPS(sites)
	write_ipeps(state,"tri_state.json", normalize=True)
	exit()
	## SUで得られた初期テンソルをリスト化する
	if Lx==1:
		SU_ten =[ [a] ]
	if Lx==2:
		#SU_ten =[ [a,b], [b,a] ]
		SU_ten =[ [a,b], [c,d] ]
	if Lx==3:
		SU_ten =[ [a,c,b], [c,b,a],[b,a,c] ]
		#SU_ten =[ [c,b,a],[b,a,c],[a,c,b]  ]
	if Lx==4:
		SU_ten =[ [a,b,a,b], [b,a,b,a],[a,b,a,b],[b,a,b,a] ]
	if Lx==6:
		SU_ten =[ [a,b,a,b,a,b], [b,a,b,a,b,a],[a,b,a,b,a,b],[b,a,b,a,b,a],[a,b,a,b,a,b],[b,a,b,a,b,a] ]

	## SUで得られた初期テンソルをクラスのリストに代入する
	for x, y in product(range(Lx), range(Ly) ):
		tensor = Tensors_CTM(SU_ten[x][y])
		CTMs[x][y] = tensor

	for i in range(maxstepCTM):
		CTMRG(CTMs, Lx, Ly)
		print("")
		## 期待値を求める ##
		for x, y in product(range(Lx), range(Ly) ):
			x1 = (x+1)%Lx ; x2 = (x+2)%Lx ; x3 = (x+3)%Lx
			y1 = (y+1)%Ly ; y2 = (y+2)%Ly ; y3 = (y+3)%Ly

			psipsi, mx, my, mz,E= ComputeQuantities(\
				SU_ten[x1][y1], SU_ten[x2][y1], SU_ten[x2][y2], SU_ten[x1][y2],\
				CTMs[x1][y1].A, CTMs[x2][y1].A, CTMs[x2][y2].A, CTMs[x1][y2].A,\
				CTMs[x][y].C1, CTMs[x3][y].C2, CTMs[x3][y3].C3, CTMs[x][y3].C4,\
				CTMs[x][y2].T4, CTMs[x][y1].T4, CTMs[x1][y].T1, CTMs[x2][y].T1,\
				CTMs[x3][y1].T2, CTMs[x3][y2].T2, CTMs[x2][y3].T3, CTMs[x1][y3].T3,Ham)

			#print(i, mx, my, mz, E, np.sqrt(mx**2+my**2+mz**2))
			print('{0},{1} {2:+4f}, {3:4f}, {4:+4f}, {5:4f} , {6:5f} '.format(i, (x1,y1), float(mx), float(my), float(mz),float(np.sqrt(mx**2+my**2+mz**2)), float(E)) )
		




	













