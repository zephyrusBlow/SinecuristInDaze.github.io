---
title: Robotics Basics

---

# Robotics Basics

## Rigid body description
### position
* describe the position and pose of coordinate sys. {B} in the coordinate sys. {A} to describe the rigid body at B
![rigid_body_coordinate_sys_1](https://hackmd.io/_uploads/H1-Pz0FSbx.png)
* use column vector to describe, $R = [x, y, z]^T$
![rigid_body_coordinate_sys_2](https://hackmd.io/_uploads/SJo9fRKHbe.png)
### pose
* Direction cosines: 
angle $\alpha, \beta, \gamma$ are the angles between vector R (which will be one of the axis of {B}) and the x-axis, y-axis, z-axis (axis of {A}) respectively
if $R = [x, y, z]$ and $\vert R \vert = \sqrt{x^2 + y^2 + z^2}$, $cos\alpha = \frac{x}{\vert R \vert}, cos\beta = \frac{y}{\vert R \vert}, cos\gamma = \frac{z}{\vert R \vert}$
which $cos^2\alpha+cos^2\beta+cos^2\gamma = 1$ and $\hat{r} = \frac{r}{\vert r\vert}$ = {$cos\alpha, cos\beta, cos\gamma$}

* Pose matrix:
use cosine of the direction of {B} relative to the direction of {A} to describe pose (Direction cosines)
$R^A_B = [x^A_B, y^A_B, z^A_B] = \begin{bmatrix} r^{Ax}_{Bx} & r^{Ax}_{By} & r^{Ax}_{Bz} \\ r^{Ay}_{Bx} & r^{Ay}_{By} & r^{Ay}_{Bz} \\ r^{Az}_{Bx} & r^{Az}_{By} & r^{Az}_{Bz}  \end{bmatrix}$, which $r^{Ax}_{By}$ is the vector of the y axis of {B} to the x axis of {A}


## Coordinate transformation
### translation
* $P^A = P^A_{B0} + P^B$
![tran](https://hackmd.io/_uploads/r1unAAKr-g.png)
### rotation
![rot](https://hackmd.io/_uploads/rkMuJ15rbg.png)
### rotation & translation
![rot&tran_1](https://hackmd.io/_uploads/Hk6zly9Sbx.png)
![rot&tran_2](https://hackmd.io/_uploads/HymLgkcSWx.png)



## Homogeneous transformation matrix
![homo_transform_matrix_1](https://hackmd.io/_uploads/r1ZV0l9HZe.png)
interpretation: 
* coordinate transformation
![homo_transform_matrix_2](https://hackmd.io/_uploads/SyvaCx9SZx.png)
* coordinate sys description
![homo_transform_matrix_3](https://hackmd.io/_uploads/HJeykW5Hbx.png)
![homo_transform_matrix_4](https://hackmd.io/_uploads/r1oO1WqHZx.png)
* kinematics operator
![homo_transform_matrix_5](https://hackmd.io/_uploads/Hkb0yW5H-e.png)



## Pose angles
### pose matrix & angles
* $R^A_B = [x^A_B, y^A_B, z^A_B] = \begin{bmatrix} r^{Ax}_{Bx} & r^{Ax}_{By} & r^{Ax}_{Bz} \\ r^{Ay}_{Bx} & r^{Ay}_{By} & r^{Ay}_{Bz} \\ r^{Az}_{Bx} & r^{Az}_{By} & r^{Az}_{Bz}  \end{bmatrix}$
in this pose matrix, only 3 el ements (3 angles to the 3 axes) are independent and determining the pose of a rigid body, so we use pose angles to describe the pose matrix
* common pose angles: RPY angles (aka fixed angles) & Euler angles
### RPY angles
* R: roll, rotate according to the axis pointing forward & backward
  P: pitch, rot according to the axis pointing left & right
  Y: yaw, rot according to the axis pointing up & down

### Euler angles

### Quaternion
    

## Wk1 Q
* 为什么需要用一个整体的“齐次矩阵”来表示位姿，而不是分开用旋转和位移？
齐次矩阵可以在4x4的矩阵中同时包含旋转和位移，想要达成多个变换只需要矩阵连乘就可以变换坐标系，但是如果分开用旋转和位移，还需要额外处理。而且齐次矩阵里计算会更统一的被处理。

* 如果一个点在 A 坐标系下有坐标，我们如何把它转换到 B 坐标系？
$P_A=[x_A, y_A, z_A, 1]^T$
$P_B=T^A_B x P_A = R^A_B × P_A + P^A_B$

* 多个关节坐标系之间的变换，最后是如何得到末端相对基座的位姿的？
直接把每相邻的坐标系的T_前一个to后一个连乘起来

## Robotic kinematics
### DOF (Degree of Freedom)

### FK (forward kinematics)
- provided joint angles
- solve end-effector position and pose
- only one solution
- link parameters:
    - link length $a_{i-1}$
    - link twist $\alpha_{i-1}$
- joint variables:
    - n joints, n+1 links, single DOF between links
    - $link_0$ connects to base (never moves)
    - $q_i = \begin{cases}
    \theta_i & \text{if joint is revolute} \\
    d_i & \text{if joint is prismatic}
    \end{cases}$
    - ![planarRRarm](https://hackmd.io/_uploads/BycuiJUI-l.png)


### IK (inverse kinematics)
- provided end-effector position and pose
- solve joint angles
- multiple solutions 


### Singularity
- occurs in certain robot configurations where the end-effector loses mobility in a particular direction, resulting in a rank-deficient Jacobian matrix
- leads to infinite joint velocities in IK solutions and loss of control
- common types:
    1. boundary singularity:
        robotic arm fully stretched or pulled together
    2. internal sngularity
        multiple joints are colinear or coplanar
- effect:
    - control difficulty
    - sudden joint speed spikes
    - path planning failure

### Redundancy
- occurs when the robot has more degrees of freedom than required for a task, leading to infinite IK solutions. Optimization criteria (e.g., obstacle avoidance, energy efficiency) are needed to choose an optimal solution
- common types:
    1. task-space redundancy
        e.g. let a 7DOF robotic arm performa 3D position task
    2. algorithmic redundancy
        e.g. solve with gradient projection



## Wk3 Q
- 正运动学和逆运动学的输入和输出是什么?
正运动学：输入：关节角；输出：末端位姿
逆运动学：输入：末端位姿；输出：关节角

- 一个平面两关节机械臂去到同一个目标点，可能会有几种解?为什么?
两种。一种是第一个关节角度小一点、第二个关节是正的elbow朝上的解；一种是第一个关节角度大一点、第二个关节是负的elbow朝下的解。

- 如果目标点超出了机械臂的最大工作范围，会怎样?
会触发boundary singularity，出现无解的情况，现实操作中可以设置触发后的错误保护或者对任务进行拆解、计算最近可达点、实现部分到达


## Trajectory planning
### what is trajectory planning?
* trajectory: robot's displacement, velocity, & acceleration in its motion
* 1. task planning
    plan the operation order, procedure, & task process of a robot
  2. path planning
    plan the position & pose of a robot in a task
  3. trajectory planning
    plan the trajectory (position + pose + velocity + acceleration) of a robot (considering time variable)
* types of motion
  1. point to point
    only fix the initial position and the final position
  2. continuous path motion (CP motion)
    fixed initial position and the final position, move according to a certain path
  - also, obstacles should be considered and avoided
* trajectory planning space
  1. joint space
    $q = [q_1, q_2, \dots, q_n]^T$
    express the joint variables (like rot angle $\theta$ or displacement d) as a time function, and plan its primary and secondary derivative at the same time; directly apply to motor's rot and drive's stretch
  2. Cartesian space / task space / operational space
    $x = [x, y, z, \alpha, \beta, \gamma]^T$
    express robot's end-effector's trajectory (position + pose + velocity + acceleration) as a time function, which x, y, z are position factors, and $\alpha, \beta, \gamma$ are pose factors (Euler angles); use Jacobian matrix to calculate the kinematics, and then move the joint to achieve a smooth motion
![trajectory_planning_space](https://hackmd.io/_uploads/Hkn05hoBWe.png)
### methods for trajectory planning
1. interpolation (provided constraints on trajectory knots)
 - e.g. $v_i = 0$, robot has to reach $a_1$ through certain point 1
 - parameterization the trajectory considering the constraints into a time function for the angles of every joint $q_i(t)$
 - directly apply to motor
 - drawback: not considering end-effectors' position and pose, could collide with obstacles
 - joint space: 
   - STEP1: IK describe trjectory knot in joint space
   - STEP2: approximate smooth function for each joint
    1. determine intial and final position and pose
    2. determine trajectory contraints for certain knots
    3. solve path function
   - STEP3: set the same time for each joint in every section of a path to ensure all the joints arrive at trajectory knots and final position at the same time
 - Cartisian space
   $P^B_i = T^B_E$
   $T^0_E = T^0_B T^B_E$
   $T^0_BN= T^0_B P^B_i (T^B_E)^{-1}$
   to avoid velocity discontinuity；
   ![path_blending_1](https://hackmd.io/_uploads/BJUU5uhHWl.png)
   ![path_blending_2](https://hackmd.io/_uploads/r1QZjOnSWe.png)

 - 1. linear interpolation / PTP
     $q(t) = q_{start} + \frac{t}{T}(q_{end} - q_{start})$
     - every joint moves at constant velocity
     - drawback: may experience velocity jump (acceleration = $\infty$) at initial and final positions
   2. cubic polynomial interpolation
     $q(t) = a_0 + a_1 t + a_2 t^2 + a_3 t^3$
     - guarantee velocity continuity
     - require angle and velocity at both initial and final position
   3. quintic polynomial interpolation
     $q(t) = a_0 + a_1 t + a_2 t^2 + a_3 t^3 + a_4 t^4 + a_5 t^5$
     - guarantee acceleration continuity (smooth motion)
     - require angle, velocity, and acceleration at both initial and final position
2. (provided analyical expression for the path of motion)
 - use function approximation (like Taylor polynomial) and transform path constraints from right-angle coordinate to joint coordinate to determine the parameterization of trajectory
3. common algorithm (sampling based)
    1. PRM (Probabilistic Roadmap) 
    - discrete planning method
    - phase1: Learning Phase
        randomly pick spots in a free space and get rid of spots at obstacles; connect the points with points near; if the edge does not bump in obstacles, the edges will form a roadmap covering the whole space
      phase2: Query Phase
        put provided initial and final positions into the roadmap formed; use search algorithms (like BFS, DFS, Dijkstra) to find the best path in the roadmap
    - Multi-query: once the roadmap is formed, we can quickly perform query of different initial and fianl positions
    - application: static env, reobot needs to move multiple times in a fixed env
    2. RRT (Rapidly-exploring Random Tree) 
    - incremental planning method
    - expand from the initial position
    - pick a random spot in the space $x_{rand}$; find the nearest knot $x_{near}$ in a provided tree; expand from $x_{near}$ to $x_{rand}$ to get to a new knot $x_{new}$: if the edge from $x_{near}$ to $x_{new}$ does not bump into any obstacles, repeat the steps above, until one of the branch of the tree reach the final position
    - Single-query: create a new tree for every task; as long as time is long enough, if the path exists, RRT must find a solution
 - people often use RRT*, Bi-RRT / RRT-Connect, Informed RRT*


## Wk2 Q
* 轨迹规划时，为什么要限制速度和加速度？
限制速度是为了保证在现实的机械的能力范围内完成运动，同时保证控制的精度
限制加速度是为了达成平滑运动，防止velocity的discontinuity

* 如果规定两秒钟必须完成一次动作，应该如何安排轨迹？
使用interpolation，无论是PTP，三次或者五次多项式，将$t\in(0,2)$带入求解


## plus
*notes for study & xbotics sim2sim & thanks to UP 知知不倦智趣Talk, ROMA-LAB*