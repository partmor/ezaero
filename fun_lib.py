import numpy as np

def panel_geo(bp, T, delta, c_r, c_t, m, n, i, j):
    dy = bp/n
    
    # A-B=forward segment
    # C-D=rear segment
    # A and C are on the same chord, idem B and D. Clockwise sequence: ABDC
    y_A = -bp/2 + j*dy
    y_B = y_A + dy
    y_C, y_D = y_A, y_B

    y_pc = y_A + dy/2
    
    # slope of the quarter chord line
    p = np.tan(T)
    
    # r,s,q are the X coordinates of the quarter chord line for y_A, y_B and y_pc respectively
    ch_025 = lambda y: c_r/4 + p*abs(y)
    r = ch_025(y_A)
    s = ch_025(y_B)
    q = ch_025(y_pc)

    # chord law evaluation
    ch_y = lambda y: c_r + (c_t - c_r)*abs(2*y/bp) 
    c_AC = ch_y(y_A)
    c_BD = ch_y(y_B)
    c_PC = ch_y(y_pc)

    # division of the chord in m equal panels
    dx_AC = c_AC/m
    dx_BD = c_BD/m
    dx_PC = c_PC/m
    x_A = (r - c_AC/4) + i*dx_AC
    x_B = (s - c_BD/4) + i*dx_BD
    x_pc = (q - c_PC/4) + (i + 3/4)*dx_PC
    
    # the first term in brackets in the expressions of xA, xB and XPC is the x
    # coordinate of the leading edge of the chord that corresponds to yA, yB
    # and yPC respectively, defined by means of the quarter chord line
    x_C = x_A + dx_AC
    x_D = x_B + dx_BD
    
    x = np.array([x_A,x_B,x_D,x_C])
    y = np.array([y_A,y_B,y_D,y_C])
    z = np.tan(delta)*np.abs(y)
    
    return x, y, z

def wing_panels(bp, T, delta, c_r, c_t, m, n):
    X = np.empty((m,n,4))
    Y = np.empty((m,n,4))
    Z = np.empty((m,n,4))
    for i in range(m):
        for j in range(n):
            X[i,j,:], Y[i,j,:], Z[i,j,:] = panel_geo(bp,T,delta,c_r,c_t,m,n,i,j)
    return X, Y, Z

def steady_wing_vortex_panels(X,Y,Z,U_i,dt,alpha):
    m, n, _ = X.shape
    dxv = (X[:,:,[3,2,2,3]] - X[:,:,[0,1,1,0]])/4
    dxv[m-1,:,2:] = 0.3*U_i*dt*np.cos(alpha)
    XV = X + dxv
    
    YV = Y
    
    ZV = np.empty((m,n,4))
    ZV[:,:,[0,1]] = Z[:,:,[0,1]] + 1/4*(Z[:,:,[3,2]] - Z[:,:,[0,1]])
    ZV[:,:,[3,2]] = Z[:,:,[0,1]] + 5/4*(Z[:,:,[3,2]] - Z[:,:,[0,1]])
    
    return XV,YV,ZV

import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D

def plot_panels(X,Y,Z,elev=25,azim=-160,edge_color='k',fill_color=1,transp=0.2,ax=None):
    m, n, _ = X.shape
    bp = Y.max()*2
    new_ax = not ax
    if new_ax:
        ax = a3.Axes3D(plt.figure())
    for i in range(m):
        for j in range(n):
            vtx = np.array([X[i,j],Y[i,j],Z[i,j]]).T
            panel = a3.art3d.Poly3DCollection([vtx])
            panel.set_facecolor((0, 0, fill_color, transp))
            panel.set_edgecolor(edge_color)
            ax.add_collection3d(panel)
    if new_ax:
        limits = (-bp/1.8,bp/1.8)
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_zlim(limits)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.view_init(elev=elev, azim=azim)
    return ax