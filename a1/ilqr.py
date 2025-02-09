#!/usr/bin/env python3
import numpy as np

class iLQR:
    def __init__(self, dt=0.1, L=0.258, Q=None, R=None, Qf=None, horizon=20, max_iter=300):
        self.dt = dt
        self.L = L
        self.horizon = horizon
        self.max_iter = max_iter
        self.Q = np.diag([1.0, 1.0, 0.5]) if Q is None else Q
        self.R = np.diag([0.1, 0.1]) if R is None else R
        self.Qf = self.Q if Qf is None else Qf

    @staticmethod
    def dynamics(x, u, dt, L):
        theta = x[2]
        v = u[0]
        delta = u[1]
        x_next = np.zeros(3)
        x_next[0] = x[0] + dt * v * np.cos(theta)
        x_next[1] = x[1] + dt * v * np.sin(theta)
        x_next[2] = x[2] + dt * (v / L) * np.tan(delta)
        return x_next

    def solve(self, x0, ref_traj):
        N = self.horizon
        dt = self.dt
        L = self.L
        Q = self.Q
        R = self.R
        Qf = self.Qf
        max_iter = self.max_iter

        n, m = 3, 2
        u_seq = np.zeros((m, N))
        x_seq = np.zeros((n, N+1))
        x_seq[:, 0] = x0
        for k in range(N):
            x_seq[:, k+1] = self.dynamics(x_seq[:, k], u_seq[:, k], dt, L)

        for it in range(max_iter):
            cost = 0.0
            for k in range(N):
                dx = x_seq[:, k] - ref_traj[:, k]
                cost += dx.T @ Q @ dx + u_seq[:, k].T @ R @ u_seq[:, k]
            dx = x_seq[:, N] - ref_traj[:, N]
            cost += dx.T @ Qf @ dx

            V_x = Qf @ (x_seq[:, N] - ref_traj[:, N])
            V_xx = Qf.copy()
            k_seq = np.zeros((m, N))
            K_seq = np.zeros((m, n, N))
            diverge = False

            for k in reversed(range(N)):
                xk = x_seq[:, k]
                uk = u_seq[:, k]
                theta = xk[2]
                v = uk[0]
                delta = uk[1]

                A = np.eye(n)
                A[0, 2] = -dt * v * np.sin(theta)
                A[1, 2] = dt * v * np.cos(theta)
                B = np.zeros((n, m))
                B[0, 0] = dt * np.cos(theta)
                B[1, 0] = dt * np.sin(theta)
                B[2, 0] = dt * (1/L) * np.tan(delta)
                B[2, 1] = dt * (v / L) * (1 / np.cos(delta)**2)

                dx = xk - ref_traj[:, k]
                lx = Q @ dx
                lu = R @ uk
                lxx = Q
                luu = R
                lux = np.zeros((m, n))

                Qx = lx + A.T @ V_x
                Qu = lu + B.T @ V_x
                Qxx = lxx + A.T @ V_xx @ A
                Quu = luu + B.T @ V_xx @ B
                Qux = lux + B.T @ V_xx @ A

                reg = 1e-6 * np.eye(m)
                try:
                    Quu_inv = np.linalg.inv(Quu + reg)
                except np.linalg.LinAlgError:
                    diverge = True
                    break

                k_ff = -Quu_inv @ Qu
                K_fb = -Quu_inv @ Qux

                k_seq[:, k] = k_ff
                K_seq[:, :, k] = K_fb

                V_x = Qx + K_fb.T @ Quu @ k_ff + K_fb.T @ Qu + Qux.T @ k_ff
                V_xx = Qxx + K_fb.T @ Quu @ K_fb + K_fb.T @ Qux + Qux.T @ K_fb
                V_xx = 0.5 * (V_xx + V_xx.T)

            if diverge:
                break

            alpha = 1.0
            found = False
            for _ in range(10):
                x_new = np.zeros((n, N+1))
                u_new = np.zeros((m, N))
                x_new[:, 0] = x0
                cost_new = 0.0

                for k in range(N):
                    du = alpha * k_seq[:, k] + K_seq[:, :, k] @ (x_new[:, k] - x_seq[:, k])
                    u_new[:, k] = u_seq[:, k] + du
                    x_new[:, k+1] = self.dynamics(x_new[:, k], u_new[:, k], dt, L)
                    dx = x_new[:, k] - ref_traj[:, k]
                    cost_new += dx.T @ Q @ dx + u_new[:, k].T @ R @ u_new[:, k]
                dx = x_new[:, N] - ref_traj[:, N]
                cost_new += dx.T @ Qf @ dx

                if cost_new < cost:
                    found = True
                    u_seq, x_seq = u_new, x_new
                    break
                else:
                    alpha *= 0.5

            if not found or abs(cost - cost_new) < 1e-6:
                break

        return u_seq, x_seq