#!/usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, CirclePolygon
from matplotlib.gridspec import GridSpec  

class TrajectoryVisualizer:
    def __init__(self, X_opt, U_opt, X_actual, U_actual, obstacle_info, L=0.258, 
                 vehicle_size=(2.0/10, 1.5/10), dt=0.1):
        """
        X_opt: Optimized trajectory (3xN array)
        U_opt: Optimized controls (2xN array)
        X_actual: Executed trajectory (3xM array)
        U_actual: Executed controls (2xM array)
        obstacle_info: Tuple (x, y, radius, safety_margin)
        dt: Time step for control inputs
        """
        self.X_opt = X_opt
        self.U_opt = U_opt
        self.X_actual = X_actual
        self.U_actual = U_actual
        self.obstacle_info = obstacle_info
        self.L = L
        self.dt = dt
        self.vehicle_length, self.vehicle_width = vehicle_size
        
        # Create figure with subplots
        self.fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(3, 2, figure=self.fig)
        self.ax_map = self.fig.add_subplot(gs[0:2, :])
        self.ax_speed = self.fig.add_subplot(gs[2, 0])
        self.ax_steer = self.fig.add_subplot(gs[2, 1])
        
        self._init_plot_elements()
        self._setup_control_plots()

    def _init_plot_elements(self):
        """Initialize map plot elements"""
        # Setup main trajectory plot
        self.ax_map.set_xlabel('x [m]')
        self.ax_map.set_ylabel('y [m]')
        self.ax_map.grid(True)
        self.ax_map.axis('equal')
        
        # Draw obstacle
        obs_x, obs_y, obs_radius, safety_margin = self.obstacle_info
        self.obstacle = CirclePolygon((obs_x, obs_y), obs_radius + safety_margin, 
                                    fc='red', alpha=0.3, ec='none', label='Safety Margin')
        self.ax_map.add_patch(self.obstacle)
        self.obstacle_core = CirclePolygon((obs_x, obs_y), obs_radius, 
                                         fc='red', alpha=0.5, ec='none', label='Obstacle')
        self.ax_map.add_patch(self.obstacle_core)
        
        # Initialize trajectory lines
        self.ref_line, = self.ax_map.plot([], [], 'r--', label='Planned Path')
        self.actual_line, = self.ax_map.plot([], [], 'b-', label='Executed Path')
        
        # Initialize vehicle representations
        self.vehicle_opt = Rectangle((0, 0), self.vehicle_length, self.vehicle_width,
                                    angle=0, fc='green', ec='black', alpha=0.8)
        self.vehicle_actual = Rectangle((0, 0), self.vehicle_length, self.vehicle_width,
                                       angle=0, fc='blue', ec='black', alpha=0.8)
        self.ax_map.add_patch(self.vehicle_opt)
        self.ax_map.add_patch(self.vehicle_actual)
        
        # Set plot limits
        all_x = np.concatenate([self.X_opt[0], self.X_actual[0]])
        all_y = np.concatenate([self.X_opt[1], self.X_actual[1]])
        buffer = 5
        self.ax_map.set_xlim(np.min(all_x)-buffer, np.max(all_x)+buffer)
        self.ax_map.set_ylim(np.min(all_y)-buffer, np.max(all_y)+buffer)
        self.ax_map.legend(loc='upper right')

    def _setup_control_plots(self):
        """Initialize control input plots"""
        # Speed plot
        self.ax_speed.set_title('Control Inputs: Speed')
        self.ax_speed.set_xlabel('Time [s]')
        self.ax_speed.set_ylabel('Speed [m/s]')
        self.speed_ref_line, = self.ax_speed.plot([], [], 'r--', label='Planned')
        self.speed_actual_line, = self.ax_speed.plot([], [], 'b-', label='Actual')
        self.ax_speed.grid(True)
        self.ax_speed.legend()
        
        # Steering plot
        self.ax_steer.set_title('Control Inputs: Steering Angle')
        self.ax_steer.set_xlabel('Time [s]')
        self.ax_steer.set_ylabel('Steering [deg]')
        self.steer_ref_line, = self.ax_steer.plot([], [], 'r--', label='Planned')
        self.steer_actual_line, = self.ax_steer.plot([], [], 'b-', label='Actual')
        self.ax_steer.grid(True)
        self.ax_steer.legend()

        # Set common limits
        max_time = max(len(self.U_opt[0]), len(self.U_actual[0])) * self.dt
        self.ax_speed.set_xlim(0, max_time)
        self.ax_steer.set_xlim(0, max_time)

    def _update_animation(self, frame):
        """Update function for animation"""
        # Calculate corresponding frame indices for both trajectories
        frame_opt = min(frame, self.X_opt.shape[1]-1)
        frame_actual = min(frame, self.X_actual.shape[1]-1)
        time = frame * self.dt

        # Update planned trajectory elements
        x_opt, y_opt, theta_opt = self.X_opt[:, frame_opt]
        self.vehicle_opt.set_xy((x_opt - self.vehicle_length/2, y_opt - self.vehicle_width/2))
        self.vehicle_opt.angle = np.rad2deg(theta_opt)
        self.ref_line.set_data(self.X_opt[0, :frame_opt+1], self.X_opt[1, :frame_opt+1])

        # Update actual trajectory elements
        x_actual, y_actual, theta_actual = self.X_actual[:, frame_actual]
        self.vehicle_actual.set_xy((x_actual - self.vehicle_length/2, y_actual - self.vehicle_width/2))
        self.vehicle_actual.angle = np.rad2deg(theta_actual)
        self.actual_line.set_data(self.X_actual[0, :frame_actual+1], self.X_actual[1, :frame_actual+1])

        # Update control plots
        if frame_opt < len(self.U_opt[0]):
            self.speed_ref_line.set_data(
                np.arange(frame_opt+1)*self.dt, 
                self.U_opt[0, :frame_opt+1]
            )
            self.steer_ref_line.set_data(
                np.arange(frame_opt+1)*self.dt, 
                np.rad2deg(self.U_opt[1, :frame_opt+1])
            )

        if frame_actual < len(self.U_actual[0]):
            self.speed_actual_line.set_data(
                np.arange(frame_actual+1)*self.dt, 
                self.U_actual[0, :frame_actual+1]
            )
            self.steer_actual_line.set_data(
                np.arange(frame_actual+1)*self.dt, 
                np.rad2deg(self.U_actual[1, :frame_actual+1])
            )

        return (self.vehicle_opt, self.vehicle_actual, 
                self.ref_line, self.actual_line,
                self.speed_ref_line, self.speed_actual_line,
                self.steer_ref_line, self.steer_actual_line)

    def create_gif(self, filename='trajectory.gif', fps=20):
        """Create and save the animation as GIF"""
        total_frames = max(self.X_opt.shape[1], self.X_actual.shape[1])
        
        ani = animation.FuncAnimation(
            self.fig,
            self._update_animation,
            frames=total_frames,
            interval=1000/fps,
            blit=True
        )
        
        ani.save(filename, writer='pillow', fps=fps)
        plt.close(self.fig)
        print(f"Saved animation to {filename}")
    