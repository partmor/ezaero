import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


class Steady_VLM:

    def set_wing_parameters(self, bp, theta, delta, c_r, c_t):
        self.bp = bp
        self.theta = theta
        self.delta = delta
        self.c_r = c_r
        self.c_t = c_t

    def set_mesh_parameters(self, m, n):
        self.m = m
        self.n = n

    def set_flight_conditions(self, U_i, alpha, rho):
        self.U_i = U_i
        self.alpha = alpha
        self.rho = rho

    def build_panel(self, i, j):
        m, n, c_r, c_t, bp = self.m, self.n, self.c_r, self.c_t, self.bp
        theta, delta = self.theta, self.delta

        dy = bp / n
        # A-B=forward segment
        # C-D=rear segment
        # A and C are on the same chord, idem B and D. Clockwise sequence: ABDC
        y_A = -bp / 2 + j * dy
        y_B = y_A + dy
        y_C, y_D = y_A, y_B

        y_pc = y_A + dy / 2

        # slope of the quarter chord line
        p = np.tan(theta)

        # r,s,q are the X coordinates of the quarter chord line for y_A, y_B
        # and y_pc respectively
        ch_025 = lambda y: c_r / 4 + p * abs(y)
        r = ch_025(y_A)
        s = ch_025(y_B)
        q = ch_025(y_pc)

        # chord law evaluation
        ch_y = lambda y: c_r + (c_t - c_r) * abs(2 * y / bp)
        c_AC = ch_y(y_A)
        c_BD = ch_y(y_B)
        c_PC = ch_y(y_pc)

        # division of the chord in m equal panels
        dx_AC = c_AC / m
        dx_BD = c_BD / m
        dx_PC = c_PC / m
        x_A = (r - c_AC / 4) + i * dx_AC
        x_B = (s - c_BD / 4) + i * dx_BD

        x_pc = (q - c_PC / 4) + (i + 3 / 4) * dx_PC

        # the first term in brackets in the expressions of xA, xB and XPC is the x
        # coordinate of the leading edge of the chord that corresponds to yA, yB
        # and yPC respectively, defined by means of the quarter chord line
        x_C = x_A + dx_AC
        x_D = x_B + dx_BD

        x = np.array([x_A, x_B, x_D, x_C])
        y = np.array([y_A, y_B, y_D, y_C])
        z = np.tan(delta) * np.abs(y)

        z_pc = np.tan(delta) * np.abs(y_pc)
        v_pc = np.array([x_pc, y_pc, z_pc])

        return np.vstack((x, y, z)).T, v_pc

    def build_wing_panels_and_collocation_points(self):
        m, n = self.m, self.n

        X = np.empty((m, n, 4, 3))
        PC = np.empty((m, n, 3))
        for i in range(m):
            for j in range(n):
                X[i, j], PC[i, j] = self.build_panel(i, j)
        self.X = X
        self.PC = PC

    def build_wing_vortex_panels(self):
        m, n = self.m, self.n
        Xc = self.X

        X = Xc[:, :, :, 0]
        Y = Xc[:, :, :, 1]
        Z = Xc[:, :, :, 2]

        dxv = (X[:, :, [3, 2, 2, 3]] - X[:, :, [0, 1, 1, 0]]) / 4
        XV = X + dxv

        YV = Y

        ZV = np.empty((m, n, 4))
        ZV[:, :, [0, 1]] = Z[:, :, [0, 1]] + 1 / \
            4 * (Z[:, :, [3, 2]] - Z[:, :, [0, 1]])
        ZV[:, :, [3, 2]] = Z[:, :, [0, 1]] + 5 / \
            4 * (Z[:, :, [3, 2]] - Z[:, :, [0, 1]])

        self.XV = np.stack([XV, YV, ZV], axis=3)

    def calc_panel_normal_vectors(self):
        m, n = self.m, self.n
        X = self.X

        d1 = X[:, :, 2] - X[:, :, 0]
        d2 = X[:, :, 1] - X[:, :, 3]
        nv = np.cross(d1, d2)
        self.N = nv / np.linalg.norm(nv, ord=2, axis=2).reshape(m, n, 1)

    def calc_wing_planform_surface(self):
        X = self.X

        x = X[:, :, :, 0]
        y = X[:, :, :, 1]
        # shoelace formula to calculate flat polygon area
        einsum_str = 'ijk,ijk->ij'
        d1 = np.einsum(einsum_str, x, np.roll(y, 1, axis=2))
        d2 = np.einsum(einsum_str, y, np.roll(x, 1, axis=2))
        self.panel_surface = 0.5 * np.abs(d1 - d2)

    def build_steady_wake(self, nb=300):
        m, n, bp = self.m, self.n, self.bp
        alpha = self.alpha
        XV = self.XV

        X_wake = np.empty((n, 4, 3))
        for j in range(n):
            X_wake[j, [0, 1]] = XV[m - 1, j, [3, 2]]
            X_wake[j, [3, 2]] = XV[m - 1, j, [3, 2]] + nb * \
                bp * np.array([np.cos(alpha), 0, np.sin(alpha)])
        self.XW = X_wake

    def biot_savart_aux(self, r1, G=1):
        r2 = np.roll(r1, shift=-1, axis=2)
        cp = np.cross(r1, r2)
        norm = lambda v: np.linalg.norm(v, ord=2, axis=3)[:, :, :, np.newaxis]
        d1 = r2 - r1
        d2 = r1 / norm(r1) - r2 / norm(r2)
        vel = -1 / (4 * np.pi) * cp / (norm(cp)**2) * \
            np.einsum('ijkl,ijkl->ijk', d1, d2)[:, :, :, np.newaxis]
        return vel.sum(axis=2)

    def calc_wing_influence_matrix(self):
        m, n = self.m, self.n
        X, XV, PC, N = self.X, self.XV, self.PC, self.N

        r = XV.reshape(m * n, 4, 3)[:, np.newaxis, :, :] - \
            PC.reshape(m * n, 3)[np.newaxis, :, np.newaxis, :]
        vel = self.biot_savart_aux(r)
        nv = N.reshape(m * n, 3)[np.newaxis, :, :]
        return np.einsum('ijk,ijk->ij', vel, nv).T

    def calc_wake_contrib_to_wing_influence_matrix(self):
        m, n = self.m, self.n
        PC, XW, N = self.PC, self.XW, self.N

        aic_w = np.zeros((m * n, m * n))
        r = XW[:, np.newaxis, :, :] - \
            PC.reshape(m * n, 3)[np.newaxis, :, np.newaxis, :]
        vel = self.biot_savart_aux(r)
        nv = N.reshape(m * n, 3)
        aic_w[:, -n:] = np.einsum('ijk,ijk->ij', vel, nv[np.newaxis, :, :]).T
        return aic_w

    def calc_influence_matrix(self):
        self.aic = self.calc_wing_influence_matrix(
        ) + self.calc_wake_contrib_to_wing_influence_matrix()

    def calc_steady_rhs(self):
        m, n = self.m, self.n
        U_i, alpha = self.U_i, self.alpha
        N = self.N

        U = U_i * np.array([np.cos(alpha), 0, np.sin(alpha)])
        self.rhs = -np.dot(N.reshape(m * n, -1), U)

    def solve_net_panel_circulation_distribution(self):
        m, n = self.m, self.n
        X, XV, PC = self.X, self.XV, self.PC
        U_i, alpha = self.U_i, self.alpha
        aic, rhs = self.aic, self.rhs

        g = np.linalg.solve(aic, rhs).reshape(m, n)

        net_g = np.empty_like(g)
        net_g[0, :] = g[0, :]
        net_g[1:, :] = g[1:, :] - g[:-1, :]
        self.net_circulation = net_g

    def aerodynamic_steady_distributions(self):
        m, n, bp = self.m, self.n, self.bp
        X, XV, PC = self.X, self.XV, self.PC
        U_i, alpha, rho = self.U_i, self.alpha, self.rho
        net_g = self.net_circulation
        S = self.panel_surface

        self.dL = net_g * rho * U_i * bp / n
        self.dp = self.dL / S
        self.cl = self.dp / (0.5 * rho * U_i**2)
        self.cl_wing = self.dL.sum() / (0.5 * rho * U_i**2 * S.sum())
        self.cl_span = self.cl.sum(axis=0) / m

    def run(self):
        self.build_wing_panels_and_collocation_points()
        self.build_wing_vortex_panels()
        self.build_steady_wake()

        self.calc_panel_normal_vectors()
        self.calc_wing_planform_surface()

        self.calc_influence_matrix()
        self.calc_steady_rhs()
        self.solve_net_panel_circulation_distribution()

        self.aerodynamic_steady_distributions()

    def get_normalized_spanwise_cl(self):
        x_values = self.PC[0, :, 1] * 2 / self.bp
        y_values = self.cl_span / self.cl_wing
        return x_values, y_values

    def plot_normalized_spanwise_cl(self):
        x_values, y_values = self.get_normalized_spanwise_cl()
        plt.figure()
        plt.plot(x_values, y_values)
        plt.xlim(0, 1)
        plt.ylim(0, max(y_values) + 0.2)
        plt.xlabel('2y/b')
        plt.ylabel('cl(y)/CL')
        plt.grid()
        ax = plt.gca()
        return ax

    def plot_panels(self, X_str, elev=25, azim=-160, edge_color='k', fill_color=1, transp=0.2, ax=None):
        m, n, bp = self.m, self.n, self.bp
        X_coord = self.X if X_str == 'X' else self.XV
        X = X_coord[:, :, :, 0]
        Y = X_coord[:, :, :, 1]
        Z = X_coord[:, :, :, 2]
        new_ax = not ax
        if new_ax:
            ax = a3.Axes3D(plt.figure())
        for i in range(m):
            for j in range(n):
                vtx = np.array([X[i, j], Y[i, j], Z[i, j]]).T
                panel = a3.art3d.Poly3DCollection([vtx])
                panel.set_facecolor((0, 0, fill_color, transp))
                panel.set_edgecolor(edge_color)
                ax.add_collection3d(panel)
        if new_ax:
            limits = (-bp / 1.8, bp / 1.8)
            ax.set_xlim(limits)
            ax.set_ylim(limits)
            ax.set_zlim(limits)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.view_init(elev=elev, azim=azim)
        return ax

    def plot_control_points(self, ax):
        PC = self.PC
        ax.scatter(xs=PC[:, :, 0].ravel(), ys=PC[
                   :, :, 1].ravel(), zs=PC[:, :, 2].ravel())
        return ax

    def plot_mesh(self, elev=25, azim=-160, ax=None):
        if not (hasattr(self, 'X') and hasattr(self, 'XV')):
            self.build_wing_panels_and_collocation_points()
            self.build_wing_vortex_panels()
        ax = self.plot_panels('X', elev=elev, azim=azim, ax=ax)
        self.plot_panels('XW', elev=elev, azim=azim,
                         edge_color='r', fill_color=0, ax=ax)
        self.plot_control_points(ax)
