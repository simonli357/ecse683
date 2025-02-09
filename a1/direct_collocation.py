#!/usr/bin/env python3
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

class TrajectoryOptimizer:
    @staticmethod
    def vehicle_dynamics_kinematic(x, u, L):
        theta = x[2]
        v = u[0]
        delta = u[1]
        x_dot = v * ca.cos(theta)
        y_dot = v * ca.sin(theta)
        theta_dot = v / L * ca.tan(delta)
        return ca.vertcat(x_dot, y_dot, theta_dot)

    def __init__(self, start_state, goal_state,
                 obstacle, N=100, T=10.0, L=0.258, safety_margin=0.12, v_bounds=(-1.0, 1.0), delta_max=np.deg2rad(20.5)):
        self.N = N
        self.T = T
        self.dt = T / N
        self.L = L
        self.start_state = start_state
        self.goal_state = goal_state
        self.obstacle = obstacle
        self.safety_margin = safety_margin
        self.v_min, self.v_max = v_bounds
        self.delta_max = delta_max

        self.opti = ca.Opti()
        self.X = self.opti.variable(3, N+1)
        self.U = self.opti.variable(2, N+1)
        self.X_opt = None
        self.U_opt = None

        self._setup_problem()

    def _setup_problem(self):
        self._setup_dynamics_constraints()
        self._setup_boundary_conditions()
        self._setup_path_constraints()
        self._setup_control_constraints()
        self._setup_objective()
        self._set_initial_guesses()

    def _setup_dynamics_constraints(self):
        for k in range(self.N):
            x_k = self.X[:, k]
            x_next = self.X[:, k+1]
            u_k = self.U[:, k]
            u_next = self.U[:, k+1]

            f_k = self.vehicle_dynamics_kinematic(x_k, u_k, self.L)
            f_next = self.vehicle_dynamics_kinematic(x_next, u_next, self.L)

            x_next_est = x_k + (self.dt / 2.0) * (f_k + f_next)
            self.opti.subject_to(x_next == x_next_est)

    def _setup_boundary_conditions(self):
        x_init, y_init, theta_init = self.start_state
        x_goal, y_goal, theta_goal = self.goal_state
        self.opti.subject_to(self.X[:, 0] == ca.vertcat(x_init, y_init, theta_init))
        self.opti.subject_to(self.X[:, self.N] == ca.vertcat(x_goal, y_goal, theta_goal))

    def _setup_path_constraints(self):
        obs_x, obs_y, obs_radius = self.obstacle
        for k in range(self.N+1):
            dist_sq = (self.X[0, k] - obs_x)**2 + (self.X[1, k] - obs_y)**2
            self.opti.subject_to(dist_sq >= (obs_radius + self.safety_margin)**2)

    def _setup_control_constraints(self):
        for k in range(self.N+1):
            self.opti.subject_to(self.U[0, k] >= self.v_min)
            self.opti.subject_to(self.U[0, k] <= self.v_max)
            self.opti.subject_to(self.U[1, k] >= -self.delta_max)
            self.opti.subject_to(self.U[1, k] <= self.delta_max)

    def _setup_objective(self):
        objective = 0
        for k in range(self.N+1):
            objective += self.U[0, k]**2 + self.U[1, k]**2
        self.opti.minimize(objective)

    def _set_initial_guesses(self):
        x_init, y_init, theta_init = self.start_state
        x_goal, y_goal, theta_goal = self.goal_state
        self.opti.set_initial(self.X[0, :], np.linspace(x_init, x_goal, self.N+1))
        self.opti.set_initial(self.X[1, :], np.linspace(y_init, y_goal, self.N+1))
        self.opti.set_initial(self.X[2, :], np.linspace(theta_init, theta_goal, self.N+1))
        self.opti.set_initial(self.U[0, :], 5.0)
        self.opti.set_initial(self.U[1, :], 0.0)

    def solve(self, solver_options=None):
        if solver_options is None:
            solver_options = {"ipopt.print_level": 0, "print_time": 0}
        self.opti.solver('ipopt', solver_options)
        sol = self.opti.solve()
        self.X_opt = sol.value(self.X)
        self.U_opt = sol.value(self.U)
        return self.X_opt, self.U_opt

    def plot_trajectory(self):
        if self.X_opt is None:
            raise ValueError("Solve the optimization problem first.")
        
        plt.figure(figsize=(8, 6))
        plt.plot(self.X_opt[0, :], self.X_opt[1, :], 'b.-', label='Optimal Trajectory')
        obs_x, obs_y, obs_radius = self.obstacle
        obstacle = plt.Circle((obs_x, obs_y), obs_radius + self.safety_margin, 
                             color='r', alpha=0.3, label='Safety Margin')
        plt.gca().add_patch(obstacle)
        obstacle_core = plt.Circle((obs_x, obs_y), obs_radius, color='r', alpha=0.5, label='Obstacle')
        plt.gca().add_patch(obstacle_core)
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.title('Optimized Trajectory')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()

