#!/usr/bin/env python3
# coding: UTF-8

import carla
import time
import numpy as np
import math
import argparse
import os
import sys
import yaml
import scipy.linalg
import casadi as ca
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosModel

##########################
# Define the Path class  #
##########################
class Path:
    """
    A helper class that converts a list of CARLA waypoints into the reference arrays
    required by the MPC controller.
    """
    def __init__(self, waypoints, v_ref=5.0):
        self.waypoints = waypoints
        self.num_waypoints = len(waypoints)
        self.waypoints_x = np.array([wp.transform.location.x for wp in waypoints])
        self.waypoints_y = np.array([wp.transform.location.y for wp in waypoints])
        # Compute state references: [x, y, psi] where psi is computed from the heading
        state_refs = []
        for i in range(len(waypoints)):
            loc = waypoints[i].transform.location
            x = loc.x
            y = loc.y
            if i < len(waypoints) - 1:
                next_loc = waypoints[i+1].transform.location
                yaw = math.atan2(next_loc.y - y, next_loc.x - x)
            else:
                yaw = state_refs[-1][2] if state_refs else 0.0
            state_refs.append([x, y, yaw])
        self.state_refs = np.array(state_refs)
        # Input references: constant desired speed and zero steering angle.
        self.input_refs = np.zeros((len(waypoints), 2))
        self.input_refs[:, 0] = v_ref
        # For waypoint normals (used for visualization or error computation), compute a perpendicular vector.
        self.wp_normals = np.array([[-math.sin(psi), math.cos(psi)] for (_, _, psi) in self.state_refs])
        # Curvature: for simplicity, we set it to zero along the path.
        self.kappa = np.zeros(len(waypoints))
        # Estimate density (waypoints per meter) and total path length (used in MPC planning).
        if len(waypoints) > 1:
            dists = np.linalg.norm(np.diff(self.state_refs[:, :2], axis=0), axis=1)
            total_length = np.sum(dists)
            self.density = len(waypoints) / total_length if total_length > 0 else 1.0
            self.rdb_circumference = total_length
        else:
            self.density = 1.0
            self.rdb_circumference = 0.0

#######################################
# Your MPC Optimizer (modified) class #
#######################################
# Here we modify your existing Optimizer to accept a "path" object.
class Optimizer(object):
    def __init__(self, x0=None):
        self.solver, self.integrator, self.T, self.N, self.t_horizon = self.create_solver()

        # The following fields are set from self.path. In the original code, self.path was assumed to exist.
        self.waypoints_x = self.path.waypoints_x
        self.waypoints_y = self.path.waypoints_y
        self.num_waypoints = self.path.num_waypoints
        self.wp_normals = self.path.wp_normals
        self.kappa = self.path.kappa
        self.density = self.path.density
        self.state_refs = self.path.state_refs
        self.input_refs = self.path.input_refs
        self.waypoints_x = self.state_refs[:, 0]
        self.waypoints_y = self.state_refs[:, 1]

        self.counter = 0
        self.target_waypoint_index = 0
        self.last_waypoint_index = 0
        self.count = 0
        density = 1 / abs(self.v_ref) / self.T
        self.region_of_acceptance = 0.05 / 10 * density * 2 * 1.5
        self.last_u = None
        self.t0 = 0
        self.init_state = x0 if x0 is not None else self.state_refs[0]
        if len(self.init_state) == 2:
            self.init_state = np.array([self.init_state[0], self.init_state[1], self.state_refs[0, 2]])
        self.update_current_state(self.init_state[0], self.init_state[1], self.init_state[2])
        self.init_state = self.current_state.copy()
        self.u0 = np.zeros((self.N, 2))
        self.next_trajectories = np.tile(self.init_state, self.N + 1).reshape(self.N + 1, -1)
        self.next_controls = np.zeros((self.N, 2))
        self.next_states = np.zeros((self.N + 1, 3))
        self.x_c = []  # history of states
        self.u_c = []
        self.u_cc = []
        self.t_c = [self.t0]  # time history
        self.xx = []
        self.x_refs = []
        self.x_errors = []
        self.y_errors = []
        self.yaw_errors = []
        # start MPC iteration counter
        self.mpciter = 0
        self.start_time = time.time()
        self.index_t = []

    def create_solver(self):
        config_path = 'config/mpc_config.yaml'
        current_path = os.path.dirname(os.path.realpath(__file__))
        path_config = os.path.join(current_path, config_path)
        with open(path_config, 'r') as f:
            config = yaml.safe_load(f)
            
        model = AcadosModel()
        # Control inputs: velocity v and steering delta.
        v = ca.SX.sym('v')
        delta = ca.SX.sym('delta')
        controls = ca.vertcat(v, delta)
        # Model states: x, y, psi.
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        psi = ca.SX.sym('psi')
        states = ca.vertcat(x, y, psi)
        # Vehicle geometry parameters.
        L_f = config['l_f']
        L_r = config['l_r']
        self.L = config['wheelbase']
        beta = ca.atan((L_r / self.L) * ca.tan(delta))
        x_dot = v * ca.cos(psi + beta)
        y_dot = v * ca.sin(psi + beta)
        psi_dot = (v * ca.cos(beta) / self.L) * ca.tan(delta)
        rhs = [x_dot, y_dot, psi_dot]

        f = ca.Function('f', [states, controls], [ca.vcat(rhs)], ['state', 'control_input'], ['rhs'])
        x_dot_sym = ca.SX.sym('x_dot', len(rhs))
        f_impl = x_dot_sym - f(states, controls)

        model.f_expl_expr = f(states, controls)
        model.f_impl_expr = f_impl
        model.x = states
        model.xdot = x_dot_sym
        model.u = controls
        model.p = ca.SX.sym('p', 2)  # parameters (e.g., previous control input)
        delta_u = controls - model.p

        model.name = config['name']
        T = config['T']
        N = config['N']
        constraint_name = 'constraints'
        cost_name = 'costs'
        t_horizon = T * N

        # constraints and cost parameters from config
        self.v_max = config[constraint_name]['v_max']
        self.v_min = config[constraint_name]['v_min']
        self.delta_max = config[constraint_name]['delta_max']
        self.delta_min = config[constraint_name]['delta_min']
        self.x_min = config[constraint_name]['x_min']
        self.x_max = config[constraint_name]['x_max']
        self.y_min = config[constraint_name]['y_min']
        self.y_max = config[constraint_name]['y_max']
        self.v_ref = config[constraint_name]['v_ref']
        self.x_cost = config[cost_name]['x_cost']
        self.y_cost = config[cost_name]['y_cost']
        self.yaw_cost = config[cost_name]['yaw_cost']
        self.v_cost = config[cost_name]['v_cost']
        self.steer_cost = config[cost_name]['steer_cost']
        self.delta_v_cost = config[cost_name]['delta_v_cost']
        self.delta_steer_cost = config[cost_name]['delta_steer_cost']
        Q = np.array([[self.x_cost, 0.0, 0.0],
                      [0.0, self.y_cost, 0.0],
                      [0.0, 0.0, self.yaw_cost]])
        R = np.array([[self.v_cost, 0.0], [0.0, self.steer_cost]])

        nx = model.x.size()[0]
        nu = model.u.size()[0]
        n_params = model.p.shape[0]

        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        acados_source_path = os.environ['ACADOS_SOURCE_DIR']
        sys.path.insert(0, acados_source_path)
        ocp = AcadosOcp()
        ocp.acados_include_path = acados_source_path + '/include'
        ocp.acados_lib_path = acados_source_path + '/lib'
        ocp.model = model
        ocp.dims.N = N
        ocp.solver_options.tf = t_horizon
        ocp.dims.np = n_params
        ocp.parameter_values = np.zeros(n_params)

        ocp.model.cost_y_expr = ca.vertcat(states, controls, delta_u)
        ocp.model.cost_y_expr_e = states
        
        ocp.cost.cost_type = 'NONLINEAR_LS'
        ocp.cost.cost_type_e = 'NONLINEAR_LS'
        S = np.diag([self.delta_v_cost, self.delta_steer_cost])
        ocp.cost.W = scipy.linalg.block_diag(Q, R, S)
        ocp.cost.W_e = Q

        ocp.constraints.lbu = np.array([self.v_min, self.delta_min])
        ocp.constraints.ubu = np.array([self.v_max, self.delta_max])
        ocp.constraints.idxbu = np.array([0, 1])
        ocp.constraints.lbx = np.array([self.x_min, self.y_min])
        ocp.constraints.ubx = np.array([self.x_max, self.y_max])
        ocp.constraints.idxbx = np.array([0, 1])

        x_ref = np.zeros(nx)
        u_ref = np.zeros(nu)
        ocp.constraints.x0 = x_ref
        ocp.cost.yref = np.concatenate((x_ref, u_ref, np.zeros(2))) 
        ocp.cost.yref_e = x_ref

        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.print_level = 0
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'

        json_file = os.path.join('./' + model.name + '_acados_ocp.json')
        solver = AcadosOcpSolver(ocp, json_file=json_file)
        integrator = AcadosSimSolver(ocp, json_file=json_file)
        return solver, integrator, T, N, t_horizon

    def update_and_solve(self):
        self.target_waypoint_index = self.find_next_waypoint()
        idx = self.target_waypoint_index
        self.next_trajectories = self.state_refs[idx:idx + self.N + 1]
        self.next_controls = self.input_refs[idx:idx + self.N]
        xs = self.state_refs[idx]
        self.solver.set(self.N, 'yref', xs)
        if self.last_u is None:
            self.last_u = np.zeros(2)
        for j in range(self.N):
            if self.target_waypoint_index + j >= self.state_refs.shape[0]:
                yref = np.concatenate((self.state_refs[-1], np.zeros(2), np.zeros(2)))
            else:
                yref = np.concatenate((self.next_trajectories[j], self.next_controls[j], np.zeros(2)))
            self.solver.set(j, 'yref', yref)
            self.solver.set(j, 'p', self.last_u)
        self.last_u[0] = self.next_controls[0, 0]
        self.solver.set(0, 'lbx', self.current_state)
        self.solver.set(0, 'ubx', self.current_state)
        status = self.solver.solve()
        if status != 0:
            print('ERROR: acados_ocp_solver returned status {}. Exiting.'.format(status))
            return None
        next_u = self.solver.get(0, 'u')
        self.counter += 1
        self.last_u = next_u
        return next_u

    def integrate_next_states(self, u_res=None):
        self.integrator.set('x', self.current_state)
        self.integrator.set('u', u_res)
        status_s = self.integrator.solve()
        if status_s != 0:
            raise Exception('acados integrator returned status {}. Exiting.'.format(status_s))
        self.current_state = self.integrator.get('x')
        self.t0 += self.T

    def update_current_state(self, x, y, yaw):
        if self.target_waypoint_index < len(self.state_refs):
            ref_yaw = self.state_refs[self.target_waypoint_index, 2]
            while yaw - ref_yaw > np.pi:
                yaw -= 2 * np.pi
            while yaw - ref_yaw < -np.pi:
                yaw += 2 * np.pi
        self.current_state = np.array([x, y, yaw])

    def find_closest_waypoint(self, current_state, min_index=-1, max_index=-1):
        min_distance_sq = np.inf
        closest = -1

        if min_index < 0:
            min_index = min(self.last_waypoint_index, len(self.state_refs) - 1)
        if max_index < 0:
            limit = int(np.floor(self.rdb_circumference / (self.v_ref * self.T)))
            max_index = min(self.target_waypoint_index + limit, len(self.state_refs) - 1)

        for i in range(max_index, min_index - 1, -1):
            diff = self.state_refs[i][:2] - current_state[:2]
            distance_sq = np.dot(diff, diff)
            if distance_sq < min_distance_sq:
                min_distance_sq = distance_sq
                closest = i

        self.closest_waypoint_index = closest
        return closest, np.sqrt(min_distance_sq)

    def find_next_waypoint(self):
        closest_idx, distance_to_current = self.find_closest_waypoint(self.current_state)

        if distance_to_current > 1.2:
            print("WARNING: distance to closest waypoint is too large:", distance_to_current)
            min_index = int(max(closest_idx - distance_to_current * self.density * 1.2, 0))
            closest_idx, distance_to_current = self.find_closest_waypoint(self.current_state, min_index, -1)

        if self.count >= 8:
            lookahead = 1  # adjust if needed based on speed
            target = closest_idx + lookahead
            self.count = 0
        else:
            target = self.target_waypoint_index + 1
            self.count += 1

        output_target = min(target, len(self.state_refs) - 1)
        self.last_waypoint_index = output_target
        self.target_waypoint_index = output_target
        return output_target

    def draw_result(self, stats, xmin=None, xmax=None, ymin=None, ymax=None, objects=None, car_states=None):
        # This function is assumed to call a plotting routine (e.g., Draw_MPC_tracking) that you have defined elsewhere.
        if xmin is None:
            xmin = self.x_min
            xmax = self.x_max
            ymin = self.y_min
            ymax = self.y_max
        Draw_MPC_tracking(self.u_c, init_state=self.init_state, 
                          robot_states=self.xx, ref_states=self.x_refs, export_fig=self.export_fig,
                          waypoints_x=self.waypoints_x, waypoints_y=self.waypoints_y, stats=stats,
                          costs=self.costs, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                          times=self.t_c, objects=objects, car_states=car_states)

    def compute_stats(self):
        print("iter:", self.mpciter)
        t_v = np.array(self.index_t)
        try:
            print("mean solve time:", t_v.mean(), "max:", t_v.max(), "min:", t_v.min(), "std:", t_v.std(), "median:", np.median(t_v))
            print((time.time() - self.start_time) / (self.mpciter))
        except:
            print("error when computing time")
        u_c = np.array(self.u_c).reshape(-1, 2)

        print("average kappa:", np.mean(self.kappa))
        average_speed = np.mean(u_c[:, 0])
        average_steer = np.mean(u_c[:, 1])
        delta_u_c = np.diff(u_c, axis=0)
        average_delta_speed = np.mean(np.abs(delta_u_c[:, 0]))
        average_delta_steer = np.mean(np.abs(delta_u_c[:, 1]))
        print(f"Average speed: {average_speed:.4f} m/s")
        print(f"Average steer angle: {average_steer:.4f} rad")
        print(f"Average change in speed: {average_delta_speed:.4f} m/s²")
        print(f"Average change in steer angle: {average_delta_steer:.4f} rad/s")
        average_x_error = np.mean(np.abs(self.x_errors))
        average_y_error = np.mean(np.abs(self.y_errors))
        self.yaw_errors = np.array(self.yaw_errors)
        self.yaw_errors = np.arctan2(np.sin(self.yaw_errors), np.cos(self.yaw_errors))
        average_yaw_error = np.mean(np.abs(self.yaw_errors))
        print(f"Average x error: {average_x_error:.4f} m")
        print(f"Average y error: {average_y_error:.4f} m")
        print(f"Average yaw error: {average_yaw_error:.4f} rad")
        stats = [average_speed, average_steer, average_delta_speed, average_delta_steer, average_x_error, average_y_error, average_yaw_error]
        return stats

# A subclass that simply injects the path into the Optimizer.
class OptimizerModified(Optimizer):
    def __init__(self, x0=None, path=None):
        if path is None:
            raise ValueError("A Path object must be provided.")
        self.path = path
        # Store the track length (used in waypoint search)
        self.rdb_circumference = self.path.rdb_circumference
        super().__init__(x0)

##########################################
# Helper functions for CARLA integration #
##########################################
def get_vehicle_state(vehicle):
    transform = vehicle.get_transform()
    x = transform.location.x
    y = transform.location.y
    yaw = math.radians(transform.rotation.yaw)
    return np.array([x, y, yaw])

def compute_speed(vehicle):
    vel = vehicle.get_velocity()
    return math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

def convert_mpc_to_carla_control(u, current_speed, delta_max):
    """
    Converts the MPC output u = [desired_speed, steering_angle] into
    CARLA VehicleControl commands.
    """
    desired_speed, delta = u
    speed_error = desired_speed - current_speed
    # Simple proportional controller for throttle/brake
    if speed_error > 0:
        throttle = np.clip(speed_error * 0.1, 0, 1)
        brake = 0.0
    else:
        throttle = 0.0
        brake = np.clip(-speed_error * 0.1, 0, 1)
    # Normalize the steering using delta_max from MPC config.
    steer = np.clip(delta / delta_max, -1, 1)
    return throttle, steer, brake

##############################
# Main script for CARLA MPC #
##############################
def main():
    argparser = argparse.ArgumentParser(description="MPC controller with CARLA integration")
    argparser.add_argument('--host', default='localhost', help='CARLA host')
    argparser.add_argument('--port', default=2000, type=int, help='CARLA port')
    args = argparser.parse_args()

    # Connect to CARLA
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle.*')[0]
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    vehicle.set_autopilot(False)
    
    # Allow some time for the simulation to settle
    time.sleep(2.0)
    
    # Retrieve waypoints from the CARLA map (using a spacing of 2 meters)
    carla_map = world.get_map()
    waypoints = carla_map.generate_waypoints(2.0)
    if len(waypoints) < 50:
        print("Not enough waypoints found.")
        vehicle.destroy()
        return
    # For this example, we simply select the first 50 waypoints.
    selected_waypoints = waypoints[:50]
    # Create the path with a desired speed (v_ref); adjust v_ref as needed.
    path = Path(selected_waypoints, v_ref=5.0)
    
    # Get initial state from the vehicle
    state = get_vehicle_state(vehicle)
    # Instantiate the MPC optimizer with the initial state and path.
    mpc = OptimizerModified(x0=state, path=path)
    
    print("Starting MPC control loop...")
    try:
        while mpc.target_waypoint_index < mpc.num_waypoints - 1:
            # Get the current vehicle state via sensor feedback.
            current_state = get_vehicle_state(vehicle)
            mpc.update_current_state(current_state[0], current_state[1], current_state[2])
            
            # Solve the MPC to obtain the next control input.
            u_res = mpc.update_and_solve()
            if u_res is None:
                print("MPC solver failed. Exiting control loop.")
                break
            
            # Get current speed from the vehicle.
            current_speed = compute_speed(vehicle)
            # Convert the MPC output to CARLA control signals.
            throttle, steer, brake = convert_mpc_to_carla_control(u_res, current_speed, mpc.delta_max)
            control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
            vehicle.apply_control(control)
            
            # Optionally tick the world if using synchronous mode and wait for next control step.
            world.tick()
            time.sleep(mpc.T)
            
            mpc.mpciter += 1  # update iteration count

    finally:
        print("Stopping vehicle and cleaning up...")
        vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        time.sleep(1)
        vehicle.destroy()

if __name__ == '__main__':
    main()
