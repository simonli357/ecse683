# ECSE 683

This repo contains the coursework for ECSE 683 at McGill University.

## Assignment 1: Optimal Control

name: Simon Li
requirements:
- Ubuntu 20.04
- ROS Noetic
- python libraries in requirements.txt

### Simulator

#### Gazebo
**Task:** Overtaking static obstacle.

For simulation environment setup, refer to [Simulator Setup Guide](https://github.com/simonli357/Simulator).

### Problem Setup

#### Task
Overtake a static vehicle on a straight line while staying within speed limits and avoiding collisions.

#### State Variables
- Car position (x, y), yaw angle θ
- Obstacle positions

#### Control Inputs
- Velocity v
- Steering angle δ

### Optimal Control Formulation

#### Cost Function
Minimize over horizon T:
```
J = Σ [w₁(v - v_desired)² + w₂(y - y_lane_center)² + w₃v² + w₄δ²] + Terminal Cost
```
**Terms:** Speed tracking, lane centering, control effort.  
**Terminal Cost:** Penalize deviation from final target position/speed.

#### Constraints

#### Vehicle Dynamics (bicycle model)
```
ẋ = v cos(θ)  
ẏ = v sin(θ)  
θ̇ = (v / L) tan(δ)  
```

#### Physical Limits
- Velocity: v ∈ [v_min, v_max]
- Steering angle: δ ∈ [δ_min, δ_max]

#### Collision Avoidance
Enforce distance_to_obstacle(t) > safety_margin for all t.

### Implementation Steps

#### Model & Simulator Setup
- Choose Gazebo. For simplicity, start with a kinematic bicycle model in Python (no simulator).
- Define state/control variables and dynamics.

#### Trajectory Optimization (Direct Collocation)
- Use CasADi (Python) for symbolic differentiation and solver setup.
- Discretize the trajectory into N steps and solve with IPOPT.

#### Inequality Constraints
- **Physical Limits:** Enforce bounds on velocity and steering angle.
- **Collision Avoidance:** Ensure vehicle does not overlap with static/dynamic obstacles.

#### Implementation with CasADi (Python)
- Use CasADi for automatic differentiation and symbolic computation.
- Define dynamics as symbolic equations.
- Solve using IPOPT, an interior-point NLP solver.

#### iLQR
- Implement iLQR with a quadratic approximation of the cost and linearized dynamics.
- Use automatic differentiation for dynamics Jacobians.

#### Integration with Simulator
- In Gazebo: Use ROS Noetic and the provided simulator instructions to spawn the ego vehicle, apply optimized controls, and track obstacles.
- Add noise to the dynamics model to test robustness.

### Additional Information

#### Examples & Configuration

- **Configuration Space:**  
  The configuration space consists of the vehicle’s full state, represented by its position (x, y) and orientation (θ). In this example, the degree of freedom is 3 (x, y, θ). The environment (task space) further includes obstacle positions and lane constraints.

- **Task Space:**  
  The task space is defined by the goal of overtaking a static obstacle while adhering to speed limits and maintaining a safe distance from obstacles. This includes the spatial boundaries and lane positions that the vehicle must respect.

- **State & Action:**  
  - **State:** The state is characterized by the vehicle’s position (x, y), its yaw angle (θ), and additional environmental factors (like obstacle positions).  
  - **Action:** The control inputs, or actions, are the velocity (v) and steering angle (δ) applied to the vehicle.

#### Method Description

- **Approach:**  
  This project uses an optimization-based approach. Two primary methods are implemented:
  - **Direct Collocation:** Utilizes CasADi for formulating and solving the optimal control problem via discretization and IPOPT.
  - **iLQR (iterative Linear Quadratic Regulator):** Implements a quadratic approximation of the cost and linearized dynamics to iteratively compute optimal controls.
  
- **Ensuring Physical Limits:**  
  Physical limits are strictly enforced as constraints within the optimization formulations. The vehicle dynamics (modeled via the kinematic bicycle model) have explicit bounds on both velocity (`v ∈ [v_min, v_max]`) and steering angle (`δ ∈ [δ_min, δ_max]`). Additionally, collision avoidance is maintained by enforcing a safety margin around obstacles.

#### Limitations

- **Performance & Robustness:**  
  While the optimization and iLQR methods provide effective control strategies for the overtaking task, the methods are not guaranteed to work 100% of the time. Limitations include:
  - **Sensitivity to Initial Conditions:** The solver's performance can be affected by the choice of initial guesses.
  - **Local Minima:** Nonlinear optimization may converge to suboptimal solutions in complex scenarios.
  - **Modeling Approximations:** The kinematic bicycle model is an approximation and may not capture all the nuances of real vehicle dynamics, especially at higher speeds or during aggressive maneuvers.
  - **Numerical Issues:** Solver convergence (both IPOPT and iLQR) can be sensitive to parameter tuning and external disturbances.
  
  These factors imply that while the method works reliably under controlled conditions, real-world uncertainties might cause occasional deviations from the optimal trajectory.

### Results

#### Optimized Trajectory (Static Visualization)
![Optimized Trajectory](a1/assets/optimized_trajectory.png)

#### Combined Trajectory (Animated)
![Combined Trajectory](a1/assets/combined_trajectory.gif)

#### Overtaking Maneuver (Gazebo Simulation)
[View Demo Video](a1/assets/a1demo.mp4)

### Example Tools/Code
- **Optimization:** CasADi (for direct collocation).
- **Baseline Code:** Modify this trajectory optimization example: [CasADi Direct Collocation Example](https://github.com/casadi/casadi/blob/main/docs/examples/python/direct_collocation.py)

### How to Use

#### Without Simulation (Only Plotting)
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the control algorithm:
   ```bash
   python3 a1/control_test.py
   ```

#### With Gazebo Simulation
1. Install ROS Noetic.
2. Follow the [Simulator Setup Guide](https://github.com/simonli357/Simulator) for environment configuration.
3. Launch Gazebo following instructions in the link.
4. Run the control node:
   ```bash
   python3 a1/control_gazebo.py
   ```

