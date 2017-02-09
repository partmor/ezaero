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
    
    z_pc = np.tan(delta)*np.abs(y_pc)
    v_pc = np.array([x_pc,y_pc,z_pc])
    
    return np.vstack((x,y,z)).T, v_pc

def wing_panels(bp, T, delta, c_r, c_t, m, n):
    X = np.empty((m,n,4,3))
    PC = np.empty((m,n,3))
    for i in range(m):
        for j in range(n):
            X[i,j], PC[i,j] = panel_geo(bp,T,delta,c_r,c_t,m,n,i,j)
    return X, PC

def steady_wing_vortex_panels(X_coord,U_i,alpha):
    X = X_coord[:,:,:,0]
    Y = X_coord[:,:,:,1]
    Z = X_coord[:,:,:,2]
    m, n = X_coord.shape[:2]
    dxv = (X[:,:,[3,2,2,3]] - X[:,:,[0,1,1,0]])/4
    # dxv[m-1,:,2:] = 0.3*U_i*dt*np.cos(alpha)
    XV = X + dxv
    
    YV = Y
    
    ZV = np.empty((m,n,4))
    ZV[:,:,[0,1]] = Z[:,:,[0,1]] + 1/4*(Z[:,:,[3,2]] - Z[:,:,[0,1]])
    ZV[:,:,[3,2]] = Z[:,:,[0,1]] + 5/4*(Z[:,:,[3,2]] - Z[:,:,[0,1]])
    
    return np.stack([XV,YV,ZV],axis=3)

def panel_normal_vectors(X):
    m = X.shape[0]
    d1 = X[:,:,2] - X[:,:,0]
    d2 = X[:,:,1] - X[:,:,3]
    nv = np.cross(d1,d2)
    return nv / np.linalg.norm(nv,ord=2,axis=2).reshape(m,-1,1)

def wing_planform_surface(X):
    x = X[:,:,:,0]
    y = X[:,:,:,1]
	# shoelace formula to calculate flat polygon area
    einsum_str = 'ijk,ijk->ij'
    d1 = np.einsum(einsum_str,x,np.roll(y,1,axis=2))
    d2 = np.einsum(einsum_str,y,np.roll(x,1,axis=2))
    return 0.5*np.abs(d1-d2)

###############################

def steady_rhs(X,alpha,U_i):
    m, n = X.shape[:2]
    U = U_i*np.array([np.cos(alpha),0,np.sin(alpha)])
    nv = panel_normal_vectors(X)
    return -np.dot(nv.reshape(m*n,-1),U)

def panel_on_pc_induced_velocity(X,PC,G=1):
    norm = lambda x: np.linalg.norm(x,ord=2,axis=1).reshape(-1,1)
    r1 = X - PC
    r2 = np.roll(r1, shift=-1, axis=0)
    
    cp = np.cross(r1,r2)
    
    d1 = r2 - r1
    d2 = r1/norm(r1) - r2/norm(r2)
    
    return (-G/(4*np.pi)*cp/(norm(cp)**2)*np.einsum('ij,ij->i', d1, d2).reshape(-1,1)).sum(axis=0)

def wing_influence_matrix(X,PC):
    m, n = X.shape[:2]
    X_r = X.reshape(m*n,4,3)
    PC_r = PC.reshape(m*n,3)
    aic = np.empty((m*n,m*n))
    nv = panel_normal_vectors(X).reshape(m*n,3)
    for r in range(m*n):
        for s in range(m*n):
            aic[r,s] = np.dot(panel_on_pc_induced_velocity(X_r[s],PC_r[r]),nv[r])
    return aic

def steady_wake(X,alpha,nb=30):
    m, n = X.shape[:2]
    bp = X[:,:,1].max()*2
    X_wake = np.empty((n,4,3))
    for j in range(n):
        X_wake[j,[0,1]] = X[m-1,j,[3,2]]
        X_wake[j,[3,2]] = X[m-1,j,[3,2]] + nb*bp*np.array([np.cos(alpha),0,np.sin(alpha)])
    return X_wake

def wake_contrib_to_wing_influence_matrix(X,PC,alpha):
    m, n = X.shape[:2]
    aic_w = np.zeros((m*n,m*n))
    X_r = X.reshape(m*n,4,3)
    PC_r = PC.reshape(m*n,3)
    nv = panel_normal_vectors(X).reshape(m*n,3)
    X_wake = steady_wake(X,alpha)
    for r in range(m*n):
        for s in range(m*n):
            if s//n == m - 1:
                aic_w[r,s] = np.dot(panel_on_pc_induced_velocity(X_wake[s%n],PC_r[r]),nv[r])
    return aic_w

def net_panel_circulation(X,PC,U_i,alpha):
    m, n = X.shape[:2]
    aic = wing_influence_matrix(X,PC) + wake_contrib_to_wing_influence_matrix(X,PC,alpha)
    rhs = steady_rhs(X,alpha,U_i)
    g = np.linalg.solve(aic,rhs).reshape(m,n)
    
    net_g = np.empty_like(g)
    net_g[0,:] = g[0,:]
    net_g[1:,:] = g[1:,:] - g[:-1,:]
    return net_g

def aerodynamic_steady_distributions(X,PC,U_i,alpha,rho):
    m,n = X.shape[:2]
    bp = X[:,:,:,1].max()*2
    net_g = net_panel_circulation(X,PC,U_i,alpha)
    dL = net_g*rho*U_i*bp/n
    S = wing_planform_surface(X)
    dp = dL/S
    cl = dp/(0.5*rho*U_i**2)
    CLw = dL.sum()/(0.5*rho*U_i**2*S.sum())
    cl_span = cl.sum(axis=0)/m
    return CLw, cl, cl_span

import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def plot_panels(X_coord,elev=25,azim=-160,edge_color='k',fill_color=1,transp=0.2,ax=None):
    X = X_coord[:,:,:,0]
    Y = X_coord[:,:,:,1]
    Z = X_coord[:,:,:,2]
    m, n = X_coord.shape[:2]
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

def plot_control_points(PC, ax):
    ax.scatter(xs=PC[:,:,0].ravel(),ys=PC[:,:,1].ravel(),zs=PC[:,:,2].ravel())
    return ax

def plot_spanwise_cl(PC,bp,cl_span,CLw):
    plt.figure()
    plt.plot(PC[0,:,1]*2/bp,cl_span/CLw)
    plt.ylim(0,max(cl_span/CLw) + 0.2)
    plt.xlim(0,1)
    plt.xlabel('2y/b')
    plt.ylabel('cl(y)/CL')
    plt.grid()

def plot_cl_distribution(PC,cl):
    fig = plt.figure()
    ax = a3.Axes3D(fig)
    surf = ax.plot_surface(PC[:,:,0],PC[:,:,1],cl,cmap=cm.coolwarm,
                           antialiased=True,shade=False,
                           cstride=1, rstride=1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('cl')
    fig.colorbar(surf, shrink=0.5, aspect=5);