import numpy as np
import math

PI = 3.1415926536

def topi(theta):
    return theta*PI/180.0
def normvec(u):
    total = np.sum(u)
    return u/total

def rotu(u, theta):
    ux, uy, uz = u
    u = np.array([ux,uy,uz]).astype(np.float)
    u = normvec(u)
    ux, uy, uz = u
    cost = math.cos(theta)
    sint = math.sin(theta)
    R = np.array([[cost + ux*ux*(1 - cost), ux*uy*(1 - cost) - uz*sint, ux*uz*(1 - cost) +uy*sint, 0],
                    [uy*ux*(1 - cost) + uz*sint, cost + uy*uy*(1 - cost), uy*uz*(1 - cost)- ux*sint, 0], 
                    [uz*ux*(1 - cost)- uy*sint, uz*uy*(1 - cost) + ux*sint, cost +uz*uz*(1 - cost), 0],
                    [0, 0, 0, 1]]).astype(np.float)
    return R

def rotx(theta):
    return rotu((1,0,0), theta)
def roty(theta):
    return rotu((0,1,0), theta)
def rotz(theta):
    return rotu((0,0,1), theta)

def transu(u):
    ux, uy, uz = u
    return np.array([[1, 0, 0, ux],
                    [0, 1, 0, uy],
                    [0, 0, 1, uz],
                    [0, 0, 0, 1]]).astype(np.float)
