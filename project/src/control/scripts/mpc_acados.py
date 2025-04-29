#!/usr/bin/env python3
# coding=UTF-8

import os
import sys

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver

import numpy as np
import scipy.linalg

import casadi as ca
from acados_template import AcadosModel
import time
import yaml
import argparse

class Optimizer(object):
    def __init__(self, path, x0 = None):
        self.path = path
        self.solver, self.integrator, self.T, self.N, self.t_horizon = self.create_solver()

        self.waypoints_x = self.path.waypoints_x
        self.waypoints_y = self.path.waypoints_y
        self.num_waypoints = self.path.num_waypoints
        self.wp_normals = self.path.wp_normals
        self.kappa = self.path.kappa
        self.density = self.path.density
        self.state_refs = self.path.state_refs
        self.input_refs = self.path.input_refs
        self.waypoints_x = self.state_refs[:,0]
        self.waypoints_y = self.state_refs[:,1]

        self.counter = 0
        self.target_waypoint_index = 0
        self.last_waypoint_index = 0
        self.count = 0
        density = 1/abs(self.v_ref)/self.T
        self.region_of_acceptance = 0.05/10*density * 2*1.5
        self.last_u = None
        self.t0 = 0
        self.init_state = x0 if x0 is not None else self.state_refs[0]
        if len(self.init_state) == 2:
            self.init_state = np.array([self.init_state[0], self.init_state[1], self.state_refs[0, 2]])
        self.update_current_state(self.init_state[0], self.init_state[1], self.init_state[2])
        self.init_state = self.current_state.copy()
        self.u0 = np.zeros((self.N, 2))
        self.next_trajectories = np.tile(self.init_state, self.N+1).reshape(self.N+1, -1) # set the initial state as the first trajectories for the robot
        self.next_controls = np.zeros((self.N, 2))
        self.next_states = np.zeros((self.N+1, 3))
        ## start MPC
        self.mpciter = 0
        self.start_time = time.time()
        self.index_t = []
        
    def create_solver(self):
        config_path = 'config/mpc_config.yaml'
        current_path = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(current_path, config_path)
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
            
        model = AcadosModel()
        # control inputs
        v = ca.SX.sym('v')
        delta = ca.SX.sym('delta')
        controls = ca.vertcat(v, delta)
        # model states
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        psi = ca.SX.sym('psi')
        states = ca.vertcat(x, y, psi)
        # Vehicle geometry parameters
        L_f = config['l_f']
        L_r = config['l_r']
        self.L = config['wheelbase']
        beta = ca.atan((L_r/self.L) * ca.tan(delta))
        x_dot   = v * ca.cos(psi + beta)
        y_dot   = v * ca.sin(psi + beta)
        psi_dot = (v * ca.cos(beta) / self.L) * ca.tan(delta)
        rhs = [x_dot, y_dot, psi_dot]

        f = ca.Function('f', [states, controls], [ca.vcat(rhs)], ['state', 'control_input'], ['rhs'])
        # acados model
        x_dot = ca.SX.sym('x_dot', len(rhs))
        f_impl = x_dot - f(states, controls)

        model.f_expl_expr = f(states, controls)
        model.f_impl_expr = f_impl
        model.x = states
        model.xdot = x_dot
        model.u = controls
        model.p = ca.SX.sym('p', 2)  # [v_prev, delta_prev]
        delta_u = controls - model.p

        model.name = config['name']
        T = config['T']
        N = config['N']
        constraint_name = 'constraints'
        cost_name = 'costs'
        t_horizon = T * N

        # constraints
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
        self.costs = np.array([self.x_cost, self.yaw_cost, self.v_cost, self.steer_cost, self.delta_v_cost, self.delta_steer_cost])
        Q = np.array([[self.x_cost, 0.0, 0.0],[0.0, self.y_cost, 0.0],[0.0, 0.0, self.yaw_cost]])*1
        R = np.array([[self.v_cost, 0.0], [0.0, self.steer_cost]])*1

        nx = model.x.size()[0]
        nu = model.u.size()[0]
        ny = nx + nu
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
        ocp.cost.yref_e = x_ref # yref_e means the reference for the last stage

        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.print_level = 0
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'

        json_file = os.path.join('./'+model.name+'_acados_ocp.json')
        solver = AcadosOcpSolver(ocp, json_file=json_file)
        integrator = AcadosSimSolver(ocp, json_file=json_file)
        return solver, integrator, T, N, t_horizon
    
    def update(self, state, v):
        self.current_state = state
        self.v_ref = v
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
            if self.target_waypoint_index+j >= self.state_refs.shape[0]:
                yref = np.concatenate((self.state_refs[-1], np.zeros(2), np.zeros(2)))
            else:
                yref = np.concatenate((self.next_trajectories[j], self.next_controls[j], np.zeros(2)))
            self.solver.set(j, 'yref', yref)
            self.solver.set(j, 'p', self.last_u)
        self.last_u[0] = self.next_controls[0, 0]
        # self.solver.set(0, 'yref', np.concatenate((self.next_trajectories[j], self.last_u, np.zeros(2))))
        self.solver.set(0, 'lbx', self.current_state)
        self.solver.set(0, 'ubx', self.current_state)
        status = self.solver.solve()
        if status != 0 :
            print('ERROR!!! acados acados_ocp_solver returned status {}. Exiting.'.format(status))
            return None
        next_u = self.solver.get(0, 'u')
        self.counter += 1
        self.last_u = next_u
        print(f"current state: {self.current_state}, target state: {self.next_trajectories[0]}, control: {next_u}")
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
            max_index = len(self.state_refs) - 1

        # Iterate from max_index down to min_index (inclusive)
        for i in range(max_index, min_index - 1, -1):
            # Compute squared distance between waypoint i and current_state (using only x and y)
            diff = self.state_refs[i][:2] - current_state[:2]
            distance_sq = np.dot(diff, diff)
            if distance_sq < min_distance_sq:
                min_distance_sq = distance_sq
                closest = i

        self.closest_waypoint_index = closest  # store for later use if needed
        return closest, np.sqrt(min_distance_sq)

    def find_next_waypoint(self):
        closest_idx, distance_to_current = self.find_closest_waypoint(self.current_state)

        if distance_to_current > 1.2:
            print("WARNING: find_next_waypoint(): distance to closest waypoint is too large:", distance_to_current)
            # Update min_index as in C++: max(closest_idx - distance_to_current * density * 1.2, 0)
            min_index = int(max(closest_idx - distance_to_current * self.density * 1.2, 0))
            closest_idx, distance_to_current = self.find_closest_waypoint(self.current_state, min_index, -1)

        if self.count >= 8:
            lookahead = 1 if self.v_ref > 0.375 else 1  # you can adjust this if needed
            target = closest_idx + lookahead
            self.count = 0
        else:
            target = self.target_waypoint_index + 1
            self.count += 1

        output_target = min(target, len(self.state_refs) - 1)
        self.last_waypoint_index = output_target
        self.target_waypoint_index = output_target  # update the target waypoint index
        return output_target
    
# if __name__ == '__main__':
#     mpc = Optimizer(None)
    
#     mpc.target_waypoint_index = 0
#     maxiter = 4000
#     print("current state: ", mpc.current_state)
#     while True:
#         if mpc.target_waypoint_index >= mpc.num_waypoints-1 or mpc.mpciter > maxiter:
#             break
#         t = time.time()
#         mpc.x_errors.append(mpc.current_state[0] - mpc.next_trajectories[0, 0])
#         mpc.y_errors.append(mpc.current_state[1] - mpc.next_trajectories[0, 1])
#         mpc.x_refs.append(mpc.next_trajectories[0, :])
#         mpc.yaw_errors.append(mpc.current_state[2] - mpc.next_trajectories[0, 2])
#         t_ = time.time()
#         u_res = mpc.update_and_solve()
#         t2 = time.time()- t_
#         if u_res is None:
#             break
#         mpc.index_t.append(t2)
#         mpc.t_c.append(mpc.t0)
#         mpc.u_c.append(u_res)
#         mpc.integrate_next_states(u_res)
#         mpc.xx.append(mpc.current_state)
#         mpc.mpciter = mpc.mpciter + 1
#     stats = mpc.compute_stats()
#     mpc.draw_result(stats, -2, 22, -2, 16)