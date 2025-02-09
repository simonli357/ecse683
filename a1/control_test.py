#!/usr/bin/env python3
import numpy as np
from ilqr import iLQR
from direct_collocation import TrajectoryOptimizer
from visualizer import TrajectoryVisualizer

if __name__ == "__main__":
    start_state=(0.0, 5.0, 0.0)
    goal_state=(5.0, 5.0, 0.0)
    obstacle=(2.0, 5.01, 0.3)
    traj_opt = TrajectoryOptimizer(start_state, goal_state, obstacle)
    X_opt, U_opt = traj_opt.solve()
    traj_opt.plot_trajectory()

    # Simulation params
    sim_time = 20.0
    dt_sim = 0.1
    N_sim = int(sim_time / dt_sim)
    horizon = 10

    # Initialize iLQR controller
    ilqr = iLQR(dt=dt_sim, L=0.258, horizon=horizon)

    # Sim loop
    x_current = np.array([0.0, 5.0, 0.0])
    traj_history = [x_current.copy()]
    control_history = []

    for t_idx in range(len(X_opt[0]) - ilqr.horizon):
        ref_horizon = X_opt[:, t_idx:t_idx+ilqr.horizon+1]
        u_seq, x_seq = ilqr.solve(x_current, ref_horizon)
        if u_seq.size == 0: break
        
        u_apply = u_seq[:, 0]
        control_history.append(u_apply)
        x_current = ilqr.dynamics(x_current, u_apply, ilqr.dt, ilqr.L)
        traj_history.append(x_current.copy())
    
    X_actual = np.array(traj_history).T
    U_actual = np.array(control_history).T
    
    obstacle_info = (*traj_opt.obstacle, traj_opt.safety_margin)
    visualizer = TrajectoryVisualizer(
        X_opt, U_opt, 
        X_actual, U_actual,
        obstacle_info,
        L=traj_opt.L,
        dt=ilqr.dt
    )
    visualizer.create_gif('combined_trajectory.gif', fps=15)

    # traj_history = np.array(traj_history).T
    # plt.figure(figsize=(8, 6))
    # plt.plot(X_opt[0], X_opt[1], 'r--', label='Reference')
    # plt.plot(traj_history[0], traj_history[1], 'b-', label='iLQR Tracking')
    # plt.xlabel('x [m]')
    # plt.ylabel('y [m]')
    # plt.legend()
    # plt.grid(True)
    # plt.axis('equal')
    # plt.show()