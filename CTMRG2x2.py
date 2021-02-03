# -*- coding: utf-8 -*-
import scipy as sp
import scipy.linalg as spl
import numpy as np
import sys
from msvd import tensor_svd
import scipy.sparse.linalg as spsl
"""
CTMおよびエッジテンソルの初期条件と添字の順番の定義

C2 --  T21 --- T22 --  C3
|       |       |       |
T12 --  A   --- B  --  T31
|       |       |       |
T11 --  D   --- C  --  T32
|       |       |       |
C1  --  T42 --- T41 --  C4 

PEPSの順番
          0          0            0        0
         /          /             |        |
    3-- a --1   3-- b --1      3--A--1  3--B--1
      / |         / |             |        |
     2  4        2  4             2        2


C2 -1       0--T21--2       0--T22--2       0- C3 
|               |1              |1              |
0                                               1

2                                               0
|                                               |
T12-1           A              B             1-T31
|                                               |
0                                               2

2                                               0
|                                               |    
T11-1           D               C            1-T32
|                                               |
0                                               2
 
1                                               0
|               |1          |1                  |
C1 -0       2--T42--0    2--T41--0           1- C4
"""


###########################################################################
def ContractPhysBond(x,conjy):
    b0 = x.shape[0]*conjy.shape[0]
    b1 = x.shape[1]*conjy.shape[1]
    b2 = x.shape[2]*conjy.shape[2]
    b3 = x.shape[3]*conjy.shape[3]
    xy = np.tensordot(x,conjy,(4,4))\
           .transpose((0,4,1,5,2,6,3,7)).reshape((b0,b1,b2,b3))
    return xy


def sandwitch(a,b,c,d,Op,A,B,C,D,Cc1,Cc2,Cc3,Cc4):
    model=1
    # impurity tensor
    impa = ContractPhysBond(np.tensordot(a,Op,(4,0)),np.conj(a))
    impb = ContractPhysBond(np.tensordot(b,Op,(4,0)),np.conj(b))
    impc = ContractPhysBond(np.tensordot(c,Op,(4,0)),np.conj(c))
    impd = ContractPhysBond(np.tensordot(d,Op,(4,0)),np.conj(d))
    # sandwitch and average
    if model ==0:
        psi_O_psi = ContractNetwork(impa,B,C,D,Cc1,Cc2,Cc3,Cc4) \
                  + ContractNetwork(A,impb,C,D,Cc1,Cc2,Cc3,Cc4) \
                  + ContractNetwork(A,B,impc,D,Cc1,Cc2,Cc3,Cc4) \
                  + ContractNetwork(A,B,C,impd,Cc1,Cc2,Cc3,Cc4)
        return psi_O_psi/4, psi_O_psi/4

    if model ==1:
        psi_Oa_psi = ContractNetwork(impa,B,C,D,Cc1,Cc2,Cc3,Cc4) 
        psi_Ob_psi = ContractNetwork(A,impb,C,D,Cc1,Cc2,Cc3,Cc4) 
        psi_Oc_psi = ContractNetwork(A,B,impc,D,Cc1,Cc2,Cc3,Cc4) 
        psi_Od_psi = ContractNetwork(A,B,C,impd,Cc1,Cc2,Cc3,Cc4)
        return (psi_Oa_psi + psi_Oc_psi)/2.,  (psi_Ob_psi + psi_Od_psi)/2.


def ContractNetwork(P,Q,R,S,Cc1,Cc2,Cc3,Cc4):
    # C2C2C2--C3C3C3
    # C2  |   |   C3
    # C2--P---Q---C3
    # |   |   |   |
    # C1--S---R---C4
    # C1  |   |   C4
    # C1C1C1--C4C4C4
    c1 = np.tensordot(Cc1,S,((1,2),(2,3)))
    c2 = np.tensordot(Cc2,P,((1,2),(3,0)))
    c3 = np.tensordot(Cc3,Q,((1,2),(0,1)))
    c4 = np.tensordot(Cc4,R,((1,2),(1,2)))
    
    c1 = np.tensordot(c1,c4,((0,3),(1,3)))
    c2 = np.tensordot(c2,c3,((1,2),(0,3)))
    return np.einsum('ijkl,ijkl',c1,c2)


def ComputeQuantities(a,b,c,d,A,B,C,D,C1,C2,C3,C4,T11,T12,T21,T22,T31,T32,T41,T42,H):

    model = 1
    # environment is contracted into four tensors
    Cc1 = np.tensordot(np.tensordot(T42,C1,(2,0)),T11,(2,0))
    Cc2 = np.tensordot(np.tensordot(T12,C2,(2,0)),T21,(2,0))
    Cc3 = np.tensordot(np.tensordot(T22,C3,(2,0)),T31,(2,0))
    Cc4 = np.tensordot(np.tensordot(T32,C4,(2,0)),T41,(2,0))
    
    # <Ψ|Ψ>
    psipsi = ContractNetwork(A,B,C,D,Cc1,Cc2,Cc3,Cc4)
    
    # <Ψ|sx|Ψ>
    sx = np.array([[0.,1.],[1.,0.]])
    mxa, mxc = sandwitch(a,b,c,d,sx,A,B,C,D,Cc1,Cc2,Cc3,Cc4)

    # <Ψ|sy|Ψ>
    sy = np.array([[0.,-1.0j],[1.0j,0.]])
    mya, myc = sandwitch(a,b,c,d,sy,A,B,C,D,Cc1,Cc2,Cc3,Cc4)
    
    # <Ψ|sz|Ψ>
    sz = np.diag([1.,-1.])
    mza, mzc = sandwitch(a,b,c,d,sz,A,B,C,D,Cc1,Cc2,Cc3,Cc4)

    #
    #print("sublattice")
    #print (np.real(mxa/psipsi), np.real(mxc/psipsi), np.real(mya/psipsi), np.real(myc/psipsi), np.real(mza/psipsi), np.real(mzc/psipsi) )

    # <Ψ|H|Ψ>
    c1 = np.tensordot(Cc1,D,((1,2),(2,3))).transpose([0,3,2,1])
    c2 = np.tensordot(Cc2,A,((1,2),(3,0))).transpose([0,3,2,1])
    c3 = np.tensordot(Cc3,B,((1,2),(0,1))).transpose([0,3,2,1])
    c4 = np.tensordot(Cc4,C,((1,2),(1,2))).transpose([0,2,3,1])
    def newshape(C):
        D1 = int(np.sqrt(C.shape[1]));  D2 = int(np.sqrt(C.shape[2]))
        return C.reshape((C.shape[0],D1,D1,D2,D2,C.shape[3]))
    d1 = np.tensordot(d,np.tensordot(newshape(Cc1),np.conj(d),([2,4],[2,3])),
                      ([2,3],[1,2])).transpose([3,1,6,0,5,4,2,7])
    d2 = np.tensordot(a,np.tensordot(newshape(Cc2),np.conj(a),([2,4],[3,0])),
                      ([0,3],[2,1])).transpose([3,1,6,0,5,4,2,7])
    d3 = np.tensordot(b,np.tensordot(newshape(Cc3),np.conj(b),([2,4],[0,1])),
                      ([0,1],[1,2])).transpose([3,1,6,0,5,4,2,7])
    d4 = np.tensordot(c,np.tensordot(newshape(Cc4),np.conj(c),([2,4],[1,2])),
                      ([1,2],[1,2])).transpose([3,0,5,1,6,4,2,7])
    def ContractWithH(da,db,cc,cd,H):

        tmp = np.tensordot(H,np.tensordot(da,np.tensordot(db,\
                    np.tensordot(newshape(cc),newshape(cd),([3,4,5],[1,2,0])),
                                                          ([3,4,5],[1,2,0])),
                                          ([0,1,2,3,4,5],[7,5,6,1,2,0])),
                           ([0,1,2,3],[0,2,1,3]))
        return tmp
    psiHpsi = ContractWithH(d1,d2,c3,c4,H)+ContractWithH(d2,d3,c4,c1,H)+\
              ContractWithH(d3,d4,c1,c2,H)+ContractWithH(d4,d1,c2,c3,H)
    return psipsi, mxa/psipsi, mza/psipsi, psiHpsi/psipsi/2.0

def ComputeQuantities_honey(a,b,c,d,A,B,C,D,C1,C2,C3,C4,T11,T12,T21,T22,T31,T32,T41,T42,Hx,Hy,Hz):

    model = 1
    # environment is contracted into four tensors
    Cc1 = np.tensordot(np.tensordot(T42,C1,(2,0)),T11,(2,0))
    Cc2 = np.tensordot(np.tensordot(T12,C2,(2,0)),T21,(2,0))
    Cc3 = np.tensordot(np.tensordot(T22,C3,(2,0)),T31,(2,0))
    Cc4 = np.tensordot(np.tensordot(T32,C4,(2,0)),T41,(2,0))

    # <Ψ|Ψ>
    psipsi = ContractNetwork(A,B,C,D,Cc1,Cc2,Cc3,Cc4)
    
    # <Ψ|sx|Ψ>
    sx = np.array([[0.,1.],[1.,0.]])*0.5
    mxa, mxc = sandwitch(a,b,c,d,sx,A,B,C,D,Cc1,Cc2,Cc3,Cc4)

    # <Ψ|sy|Ψ>
    sy = np.array([[0.,-1.0j],[1.0j,0.]])*0.5
    mya, myc = sandwitch(a,b,c,d,sy,A,B,C,D,Cc1,Cc2,Cc3,Cc4)
    
    # <Ψ|sz|Ψ>
    sz = np.diag([1.,-1.])*0.5
    mza, mzc = sandwitch(a,b,c,d,sz,A,B,C,D,Cc1,Cc2,Cc3,Cc4)

    #print np.real(mxa/psipsi), np.real(mxc/psipsi), np.real(mya/psipsi), np.real(myc/psipsi), np.real(mza/psipsi), np.real(mzc/psipsi)

    # <Ψ|H|Ψ>
    c1 = np.tensordot(Cc1,D,((1,2),(2,3))).transpose([0,3,2,1])
    c2 = np.tensordot(Cc2,A,((1,2),(3,0))).transpose([0,3,2,1])
    c3 = np.tensordot(Cc3,B,((1,2),(0,1))).transpose([0,3,2,1])
    c4 = np.tensordot(Cc4,C,((1,2),(1,2))).transpose([0,2,3,1])
    def newshape(C):
        D1 = int(np.sqrt(C.shape[1]));  D2 = int(np.sqrt(C.shape[2]))
        return C.reshape((C.shape[0],D1,D1,D2,D2,C.shape[3]))
    d1 = np.tensordot(d,np.tensordot(newshape(Cc1),np.conj(d),([2,4],[2,3])),
                      ([2,3],[1,2])).transpose([3,1,6,0,5,4,2,7])
    d2 = np.tensordot(a,np.tensordot(newshape(Cc2),np.conj(a),([2,4],[3,0])),
                      ([0,3],[2,1])).transpose([3,1,6,0,5,4,2,7])
    d3 = np.tensordot(b,np.tensordot(newshape(Cc3),np.conj(b),([2,4],[0,1])),
                      ([0,1],[1,2])).transpose([3,1,6,0,5,4,2,7])
    d4 = np.tensordot(c,np.tensordot(newshape(Cc4),np.conj(c),([2,4],[1,2])),
                      ([1,2],[1,2])).transpose([3,0,5,1,6,4,2,7])
    def ContractWithH(da,db,cc,cd,H):

        tmp = np.tensordot(H,np.tensordot(da,np.tensordot(db,\
                    np.tensordot(newshape(cc),newshape(cd),([3,4,5],[1,2,0])),
                                                          ([3,4,5],[1,2,0])),
                                          ([0,1,2,3,4,5],[7,5,6,1,2,0])),
                           ([0,1,2,3],[0,2,1,3]))
        return tmp
    psiHpsi = ContractWithH(d1,d2,c3,c4,Hy)+ContractWithH(d2,d3,c4,c1,Hx)+\
              ContractWithH(d3,d4,c1,c2,Hz)
    #print ContractWithH(d1,d2,c3,c4,H)/psipsi, ContractWithH(d2,d3,c4,c1,H)/psipsi, ContractWithH(d3,d4,c1,c2,H)/psipsi
    return np.real(psipsi), np.real(mxa/psipsi), np.real(mya/psipsi), np.real(mza/psipsi), np.real(psiHpsi/psipsi/3*1.5)

###########################################################################

def initial_CTMRG(a,b,c,d):
    # tensor C's
    C1 = np.tensordot(b,np.conj(b),((2,3,4),(2,3,4)))\
           .transpose((1,3,0,2)).reshape((b.shape[1]**2,b.shape[0]**2))
    C2 = np.tensordot(c,np.conj(c),((0,3,4),(0,3,4)))\
           .transpose((1,3,0,2)).reshape((c.shape[2]**2,c.shape[1]**2))
    C3 = np.tensordot(d,np.conj(d),((0,1,4),(0,1,4)))\
           .transpose((1,3,0,2)).reshape((d.shape[3]**2,d.shape[2]**2))
    C4 = np.tensordot(a,np.conj(a),((1,2,4),(1,2,4)))\
           .transpose((0,2,1,3)).reshape((a.shape[0]**2,a.shape[3]**2))
    '''
    C1 /= np.max(abs(C1))
    C2 /= np.max(abs(C2))
    C3 /= np.max(abs(C3))
    C4 /= np.max(abs(C4))
    '''
    # tensor T's
    T11 = np.tensordot(c,np.conj(c),((3,4),(3,4))).transpose((2,5,1,4,0,3))\
            .reshape((c.shape[2]**2,c.shape[1]**2,c.shape[0]**2))
    T12 = np.tensordot(b,np.conj(b),((3,4),(3,4))).transpose((2,5,1,4,0,3))\
            .reshape((b.shape[2]**2,b.shape[1]**2,b.shape[0]**2))
    T21 = np.tensordot(d,np.conj(d),((0,4),(0,4))).transpose((2,5,1,4,0,3))\
            .reshape((d.shape[3]**2,d.shape[2]**2,d.shape[1]**2))
    T22 = np.tensordot(c,np.conj(c),((0,4),(0,4))).transpose((2,5,1,4,0,3))\
            .reshape((c.shape[3]**2,c.shape[2]**2,c.shape[1]**2))
    T31 = np.tensordot(a,np.conj(a),((1,4),(1,4))).transpose((0,3,2,5,1,4))\
            .reshape((a.shape[0]**2,a.shape[3]**2,a.shape[2]**2))
    T32 = np.tensordot(d,np.conj(d),((1,4),(1,4))).transpose((0,3,2,5,1,4))\
            .reshape((d.shape[0]**2,d.shape[3]**2,d.shape[2]**2))
    T41 = np.tensordot(b,np.conj(b),((2,4),(2,4))).transpose((1,4,0,3,2,5))\
            .reshape((b.shape[1]**2,b.shape[0]**2,b.shape[3]**2))
    T42 = np.tensordot(a,np.conj(a),((2,4),(2,4))).transpose((1,4,0,3,2,5))\
            .reshape((a.shape[1]**2,a.shape[0]**2,a.shape[3]**2))
    '''
    T11 /= np.max(abs(T11))
    T12 /= np.max(abs(T12))
    T21 /= np.max(abs(T21))
    T22 /= np.max(abs(T22))
    T31 /= np.max(abs(T31))
    T32 /= np.max(abs(T32))
    T41 /= np.max(abs(T41))
    T42 /= np.max(abs(T42))
    '''
    return C1, C2, C3, C4, T11, T12, T21, T22, T31, T32, T41, T42


# Top--2    2 1
# | |       | |      : create an isometry for these tensors
# 0 1   ,   Bot--0
def isometry(Top,Bot,chi):
    top = np.tensordot(Top,np.conj(Top),(2,2))
    bot = np.tensordot(Bot,np.conj(Bot),(0,0)) # 2          0
    #  Top-----Top+        Bot-----Bot+         z---lambda---z+
    # 0| |1   3| |2   +   1| |0   2| |3   =   0/|1          1|\2
    tmp = top.transpose((0,1,3,2))+bot.transpose((1,0,2,3))
    z, _, zdag = tensor_svd(tmp,(0,1),(2,3),chi)
    return z, zdag

def Philippe_isometry(A,B,C,D,C1,C2,C3,C4,T11,T12,T21,T22,T31,T32,T41,T42):
    """
       3
       |
    2--A--0  必ず'move'方向の反対側が'0'になるようにする!
       |
       1
    """
    ## それぞれの四隅にまとめる
    C1_n = np.transpose(
        np.tensordot(
            D, np.tensordot(
                T11, np.tensordot(
                    C1, T42, ([0], [2])
                ), ([0], [0])
            ), ([2, 3], [3, 0])
        ), [3, 1, 2, 0]
    )

    C2_n = np.transpose(
        np.tensordot(
            A, np.tensordot(
                T21, np.tensordot(
                    C2, T12, ([0], [2])
                ), ([0], [0])
            ), ([2, 3], [3, 0])
        ), [3, 1, 2, 0]
    )

    C3_n = np.transpose(
        np.tensordot(
            B, np.tensordot(
                T22, np.tensordot(
                    C3, T31, ([1], [0])
                ), ([2], [0])
            ), ([0, 3], [2, 1])
        ), [2, 1, 3, 0]
    )

    C4_n = np.transpose(
        np.tensordot(
            C, np.tensordot(
                T32, np.tensordot(
                    C4, T41, ([1], [0])
                ), ([2], [0])
            ), ([0, 1], [1, 2])
        ), [2, 1, 3, 0]
    )

    # upper halfとlower halfを作り、QR分解する
    upper_half = np.tensordot(C3_new, C2_new, ([0,1],[2,3]) )
    lower_half = np.tensordot(C4_new, C1_new, ([2,3],[0,1]) )
    _,R_up = tensor_QR(upper_half, (0,1), (2,3), C1.shape[0]*A.shape[0])
    _,R_low = tensor_QR(lower_half, (0,1), (2,3), C1.shape[0]*A.shape[0])

    ## Projection Operatorを作る
    U,s,Vdag =psvd(np.tensordot(R_up,R_low, ( [1,2],[1,2]) ) ,chi_cut )
    U = U*(np.sqrt(1./s))[None,:]
    Vdag = Vdag*(np.sqrt(1./s))[:,None]
    P = np.tensordot(U,R_up,([0],[0]) )
    P_til = np.tensordot(Vdag,R_low,([1],[0]) )

    return P, P_til

def update(Ca,Ta1,Ta2,Cb,Tb,Tc,X,Y,chi):
    # Cb--Tb--     Cb--
    # |   |        ||         /             3
    # Ta2-X---     Ta2-     Q2--            |
    # |   |     =  ||    =  ||           2--X-- 0  必ず外側が '0' になるようにしている
    # Ta1-Y---     Ta1-     Q1--            |
    # |   |        ||         \             1
    # Ca--Tc--     Ca--
    cb = np.tensordot(Cb,Tb,(1,0))
    ta2= np.tensordot(X,Ta2,(2,1))
    ta1= np.tensordot(Y,Ta1,(2,1))
    ca = np.tensordot(Tc,Ca,(2,0))
    Q2 = np.tensordot(ta2,cb,((2,4),(1,0))).transpose((2,1,0,3))
    Q1 = np.tensordot(ca,ta1,((1,2),(1,3)))
    
    # truncation
    z, zdag = isometry(cb,ca,chi)
    #Dsq = X.shape[0]
    #w, wdag = isometry(Q2.reshape((Q2.shape[0],Dsq,Dsq*Q2.shape[3])),
    #                   Q1.reshape((Q1.shape[0]*Dsq,Dsq,Q1.shape[3])),chi)
    w, wdag = isometry(Q2.reshape((Q2.shape[0],X.shape[1],
                                   X.shape[0]*Q2.shape[3])),
                       Q1.reshape((Q1.shape[0]*Y.shape[0],
                                   Y.shape[3],Q1.shape[3])),chi)
    cb = np.tensordot(zdag,cb,((1,2),(1,0)))
    ta2= np.tensordot(np.tensordot(wdag,ta2,((1,2),(1,3))),z,((2,3),(1,0)))
    ta1= np.tensordot(np.tensordot(zdag,ta1,((1,2),(1,3))),w,((2,3),(1,0)))
    ca = np.tensordot(ca,z,((1,2),(1,0)))
    
    # normalization
    ca /= np.max(abs(ca))
    cb /= np.max(abs(cb))
    ta1/= np.max(abs(ta1))
    ta2/= np.max(abs(ta2))
    return ca, ta1, ta2, cb


def updateOneDirection(Ca,Ta1,Ta2,Cb,Tb1,Tb2,Tc1,Tc2,P,Q,R,S,chi):
    # Cb--Tb1-Tb2-       Cb--Tb2-       Cb--
    # |   |   |          |   |          |
    # Ta2-P---Q---       Ta2-Q---       Ta2-
    # |   |   |     -->  |   |     -->  |
    # Ta1-S---R---       Ta1-R---       Ta1-
    # |   |   |          |   |          |
    # Ca--Tc2-Tc1-       Ca--Tc1-       Ca--
    Ca,Ta1,Ta2,Cb = update(Ca,Ta1,Ta2,Cb,Tb1,Tc2,P,S,chi)
    Ca,Ta1,Ta2,Cb = update(Ca,Ta1,Ta2,Cb,Tb2,Tc1,Q,R,chi)
    return Ca, Ta1, Ta2, Cb

###########################################################################
## for full update  #######################################################
###########################################################################
def CreateNLR(Ca,Ta1,Ta2,Cb,Tb1,Tb2,Cc,Tc1,Tc2,Cd,Td1,Td2,X,Y,P,S):
    D0 = X.shape[0]; D1=X.shape[1];   Dsq = D1*D1
    def tenprodX(X):
        return np.tensordot(X[None,:],np.conj(X[None,:]),(0,0))\
                 .transpose((0,4,1,5,2,6,3,7)).reshape((D0,Dsq,Dsq,4*Dsq))
    def tenprodY(X):
        return np.tensordot(X[None,:],np.conj(X[None,:]),(0,0))\
                 .transpose((0,4,1,5,2,6,3,7)).reshape((Dsq,Dsq,D0,4*Dsq))
    Q = tenprodX(X);    R = tenprodY(Y)
    # Cb--Tb1-Tb2-Cc
    # |   |   |   |         __NLR__
    # Ta2-Q= =R---Tc1      /       \
    # |   |   |   |    ->  |-0   2-|
    # Ta1-P---S---Tc2      \_     _/
    # |   |   |   |          1   3
    # Ca--Td2-Td1-Cd
    Cca = np.tensordot(np.tensordot(np.tensordot(Td2,Ca,(2,0)),Ta1,(2,0)),
                       P,((1,2),(2,3)))
    Ccb = np.tensordot(np.tensordot(np.tensordot(Ta2,Cb,(2,0)),Tb1,(2,0)),
                       Q,((1,2),(1,0)))
    Ccc = np.tensordot(np.tensordot(np.tensordot(Tb2,Cc,(2,0)),Tc1,(2,0)),
                       R,((1,2),(0,1)))
    Ccd = np.tensordot(np.tensordot(np.tensordot(Tc2,Cd,(2,0)),Td1,(2,0)),
                       S,((1,2),(1,2)))
    CL = np.tensordot(Cca,Ccb,((1,2),(0,2)))
    CR = np.tensordot(Ccd,Ccc,((0,2),(1,2)))
    NLR = np.tensordot(CL,CR,((0,1,2),(0,1,2))).reshape((2*D1,2*D1,2*D1,2*D1))
    return NLR

def CreateNLR_x(Ca,Ta1,Ta2,Cb,Tb1,Tb2,Cc,Tc1,Tc2,Cd,Td1,Td2,X,Y,P,S):
    D0 = X.shape[1]; D1=X.shape[0];   Dsq = D1*D1
    def tenprodX(X):
        return np.tensordot(X[None,:],np.conj(X[None,:]),(0,0))\
                 .transpose((0,4,1,5,2,6,3,7)).reshape((Dsq,D0,Dsq,4*Dsq))
    def tenprodY(X):
        return np.tensordot(X[None,:],np.conj(X[None,:]),(0,0))\
                 .transpose((0,4,1,5,2,6,3,7)).reshape((Dsq,D0,Dsq,4*Dsq))
    Q = tenprodX(X);    R = tenprodY(Y)
    # Cb--Tb1-Tb2-Cc
    # |   |   |   |         __NLR__
    # Ta2-Q= =R---Tc1      /       \
    # |   |   |   |    ->  |-0   2-|
    # Ta1-P---S---Tc2      \_     _/
    # |   |   |   |          1   3
    # Ca--Td2-Td1-Cd
    Cca = np.tensordot(np.tensordot(np.tensordot(Td2,Ca,(2,0)),Ta1,(2,0)),
                       P,((1,2),(2,3)))
    Ccb = np.tensordot(np.tensordot(np.tensordot(Ta2,Cb,(2,0)),Tb1,(2,0)),
                       Q,((1,2),(1,0)))
    Ccc = np.tensordot(np.tensordot(np.tensordot(Tb2,Cc,(2,0)),Tc1,(2,0)),
                       R,((1,2),(0,1)))
    Ccd = np.tensordot(np.tensordot(np.tensordot(Tc2,Cd,(2,0)),Td1,(2,0)),
                       S,((1,2),(1,2)))
    CL = np.tensordot(Cca,Ccb,((1,2),(0,2)))
    CR = np.tensordot(Ccd,Ccc,((0,2),(1,2)))
    NLR = np.tensordot(CL,CR,((0,1,2),(0,1,2))).reshape((2*D1,2*D1,2*D1,2*D1))
    return NLR

def CreateNLR_y(Ca,Ta1,Ta2,Cb,Tb1,Tb2,Cc,Tc1,Tc2,Cd,Td1,Td2,X,Y,P,S):
    D0 = X.shape[2]; D1=X.shape[1];   Dsq = D1*D1
    def tenprodX(X):
        return np.tensordot(X[None,:],np.conj(X[None,:]),(0,0))\
                 .transpose((0,4,1,5,2,6,3,7)).reshape((Dsq,Dsq,D0,4*Dsq))
    def tenprodY(X):
        return np.tensordot(X[None,:],np.conj(X[None,:]),(0,0))\
                 .transpose((0,4,1,5,2,6,3,7)).reshape((D0,Dsq,Dsq,4*Dsq))
    Q = tenprodX(X);    R = tenprodY(Y)
    # Cb--Tb1-Tb2-Cc
    # |   |   |   |         __NLR__
    # Ta2-Q= =R---Tc1      /       \
    # |   |   |   |    ->  |-0   2-|
    # Ta1-P---S---Tc2      \_     _/
    # |   |   |   |          1   3
    # Ca--Td2-Td1-Cd
    Cca = np.tensordot(np.tensordot(np.tensordot(Td2,Ca,(2,0)),Ta1,(2,0)),
                       P,((1,2),(2,3)))
    Ccb = np.tensordot(np.tensordot(np.tensordot(Ta2,Cb,(2,0)),Tb1,(2,0)),
                       Q,((1,2),(1,0)))
    Ccc = np.tensordot(np.tensordot(np.tensordot(Tb2,Cc,(2,0)),Tc1,(2,0)),
                       R,((1,2),(0,1)))
    Ccd = np.tensordot(np.tensordot(np.tensordot(Tc2,Cd,(2,0)),Td1,(2,0)),
                       S,((1,2),(1,2)))
    CL = np.tensordot(Cca,Ccb,((1,2),(0,2)))
    CR = np.tensordot(Ccd,Ccc,((0,2),(1,2)))
    NLR = np.tensordot(CL,CR,((0,1,2),(0,1,2))).reshape((2*D1,2*D1,2*D1,2*D1))
    return NLR

def CreateNLR_z(Ca,Ta1,Ta2,Cb,Tb1,Tb2,Cc,Tc1,Tc2,Cd,Td1,Td2,X,Y,P,S):
    D0 = X.shape[0]; D1=X.shape[1];   Dsq = D1*D1
    def tenprodX(X):
        return np.tensordot(X[None,:],np.conj(X[None,:]),(0,0))\
                 .transpose((0,4,1,5,2,6,3,7)).reshape((D0,Dsq,Dsq,4*Dsq))
    def tenprodY(X):
        return np.tensordot(X[None,:],np.conj(X[None,:]),(0,0))\
                 .transpose((0,4,1,5,2,6,3,7)).reshape((Dsq,Dsq,D0,4*Dsq))
    Q = tenprodX(X);    R = tenprodY(Y)
    # Cb--Tb1-Tb2-Cc
    # |   |   |   |         __NLR__
    # Ta2-Q= =R---Tc1      /       \
    # |   |   |   |    ->  |-0   2-|
    # Ta1-P---S---Tc2      \_     _/
    # |   |   |   |          1   3
    # Ca--Td2-Td1-Cd
    Cca = np.tensordot(np.tensordot(np.tensordot(Td2,Ca,(2,0)),Ta1,(2,0)),
                       P,((1,2),(2,3)))
    Ccb = np.tensordot(np.tensordot(np.tensordot(Ta2,Cb,(2,0)),Tb1,(2,0)),
                       Q,((1,2),(1,0)))
    Ccc = np.tensordot(np.tensordot(np.tensordot(Tb2,Cc,(2,0)),Tc1,(2,0)),
                       R,((1,2),(0,1)))
    Ccd = np.tensordot(np.tensordot(np.tensordot(Tc2,Cd,(2,0)),Td1,(2,0)),
                       S,((1,2),(1,2)))
    CL = np.tensordot(Cca,Ccb,((1,2),(0,2)))
    CR = np.tensordot(Ccd,Ccc,((0,2),(1,2)))
    NLR = np.tensordot(CL,CR,((0,1,2),(0,1,2))).reshape((2*D1,2*D1,2*D1,2*D1))
    return NLR

def costfunction(aR,bL,NLR,abU):
    tmp = np.tensordot(NLR,abU,((0,2),(0,1)))
    T = np.einsum('ijkl,ijkl',tmp,np.conj(abU))
    Sa = np.einsum('ijk,ijk',np.tensordot(tmp,np.conj(aR),((0,2),(0,2))),
                   np.conj(bL).transpose((0,2,1)))
    aRa = np.einsum('ijk,ijk',np.conj(bL),
                    np.transpose(np.tensordot(
                        np.conj(aR),
                        np.tensordot(NLR,
                                     np.tensordot(aR, bL, (1,1)),
                                     ([0,2],[0,2])),([0,2],[0,2])),[1,0,2]))
    return T + aRa - Sa - np.conj(Sa)

########################    相関長を求めてみる    ###############################

def Corre_Length(T21,T22,T42,T41):

    def matvec(vec):
        vv = np.reshape(vec, (T21.shape[0], T21.shape[0]))
        vv = np.tensordot(
            T21, np.tensordot(
                T42, np.tensordot(
                    T22, np.tensordot(
                        T41, vv, ([0], [1])
                    ), ([1, 2], [0, 2])
                ), ([0], [1])
            ), ([1, 2], [0, 2])
        )
        return np.reshape(vv, (T21.shape[0], T21.shape[0]))

    def sort(s,vr):

        Vr = np.zeros( (vr.shape[0],vr.shape[1]), dtype=float )
        sr = np.ones(s.shape)
        idx_r = np.argsort(np.real(s))[::-1]
        for idx, p in enumerate(idx_r):
            Vr[:,idx]=np.real(vr[:,p])
            sr[idx]=np.real(s[p])
        return sr,Vr

    Tra = np.transpose(
        np.tensordot(
            np.tensordot(
                T21, T22, ([2], [0])
            ), np.tensordot(
                T42, T41, ([0], [2])
            ), ([1, 2], [0, 3])
        ), [0, 2, 1, 3]
    )
    TT = Tra.reshape(Tra.shape[0]**2, Tra.shape[0]**2)
    #print(TT,"\n\n")

    #print(TT.T)
    num = 10 # 求める固有値の数 
    lin_op = spsl.LinearOperator(
        (T21.shape[0]**2, T21.shape[0]**2), matvec=matvec)

    s,vector = spsl.eigs(lin_op, num, which='LR', return_eigenvectors=True)
    s = sorted(np.real(s), reverse=True)[:num]

    s /= np.max(s)


    length_1 = -2.0/np.log(s[1]/s[0])
    length_2 = -2.0/np.log(s[2]/s[0])
    length_3 = -2.0/np.log(s[3]/s[0])
    length_4 = -2.0/np.log(s[4]/s[0])
    length_5 = -2.0/np.log(s[5]/s[0])

    return length_1, length_2, length_3, length_4, length_5



