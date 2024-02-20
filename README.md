# group-project-controller-code

This repo houses the code for the *controller* portion of the group project. The purpose of this controller is to control an autonomous vehicle along a straight path, while avoiding being pushed off the path by a crosswind, or colliding with static obstacles.

## Design of MDP
### State
The state of the system is defined by the current position and velocity of the vehicle, the current steering angle of the vehicle, the current windspeed, and the (fixed) positions of all obstacles.
```math
s = \bigg[(p_x, p_y), (v_x,v_y), \alpha, w, O\bigg]
```

### Action
The action space is given by a continuous forward acceleration value, and a continuous change in steering angle (i.e. angular velocity); each bounded by some maximum magnitude.
```math
a = (a_x, v_\alpha) \in [-a_\text{max}, a_\text{max}] \times [-v_\alpha_\text{max}, v_\alpha_\text{max}]
```

### Transition function
The transition function is given by the following equations:
```math
\begin{align}
p_x(t+1) &= p_x(t) + v_x(t)\Delta t \\
p_y(t+1) &= p_y(t) + (v_y(t) + w(t))\Delta t \\
v_x(t+1) &= v_x(t) + a_x(t)\Delta t \\
v_y(t+1) &= v_y(t) + \sin\big(\alpha\big)\Delta t \\
\alpha(t+1) &= \alpha(t) + v_\alpha(t)\Delta t \\
w(t+1) &= w(t) + \epsilon\Delta t \\
O(t+1) &= O(t)
\end{align}
```
where $\epsilon$ is some scalar random variable representing the wind gust (we can think later about what the distribution should look like).

### Reward function
We want the reward function to reflect three (terminal) outcomes: (1) the vehicle is pushed off the path by the wind (bad), (2) the vehicle collides with an obstacle (bad), or (3) the vehicle reaches the end of the path (good). In addition to these outcomes, we penalize the vehicle at each timestep based on its distance from both the goal and the centre of the path, with an additional penalty given for being stationary over multiple timesteps. We can define the reward function as follows:
```math
r_\text{base}(s,a) = -c_1\bigg(\sqrt{(p_x - x_\text{max})^2 + p_y^2}\bigg) - c_2\|p_y|
```
```math
r(s,a) = r_\text{base}(s,a) + \begin{cases}
c_3 & \text{if } p_x \geq x_\text{max} \\
-c_3 & \text{if } p_y \notin [-y_\text{max}, y_\text{max}] \\
-c_3 & \text{if } \exists o \in O \text{ such that } \sqrt{(p_x - o_x)^2 + (p_y - o_y)^2} \leq r_\text{min} \\
-c_4 & \text{if } t_\text{stationary} \geq \tau
\end{cases}
```
where $c_1,c_2,c_3$ are constants, $y_\text{max}$ is the maximum y-coordinate of the path, $r_\text{min}$ is the minimum distance between the vehicle and an obstacle that constitutes a collision, $x_\text{max}$ is the maximum x-coordinate of the path, and $\tau$ is the threshold for penalising stationarity.
