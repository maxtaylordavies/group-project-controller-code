# group-project-controller-code

This repo houses the code for the *controller* portion of the group project. The purpose of this controller is to control an autonomous vehicle along a straight path, while avoiding being pushed off the path by a crosswind, or colliding with static obstacles.

## Design of MDP
**State**: The state of the system is defined by the position of the vehicle along the path, the velocity of the vehicle, the current windspeed, and the (fixed) positions of all obstacles.
$$
s = \{(p_x, p_y), (v_x,v_y), w, O\}
$$

**Action**: The action space is given by a pair of continuous acceleration values in x and y, bounded by some maximum acceleration value.
$$
a = (a_x, a_y) \in [-a_\text{max}, a_\text{max}]^2
$$

**Transition function**: The transition function is given by the following equations:
$$
p_x(t+1) = p_x(t) + v_x(t) \\
p_y(t+1) = p_y(t) + v_y(t) + w(t) \\
v_x(t+1) = v_x(t) + a_x(t) \\
v_y(t+1) = v_y(t) + a_y(t) \\
w(t+1) = w(t) + \epsilon
O(t+1) = O(t)
$$
where $\epsilon$ is some random variable representing the wind gust (we can think later about what the distribution should look like).

**Reward function**: The reward function essentially reflects three outcomes: (1) the vehicle is pushed off the path by the wind (bad), (2) the vehicle collides with an obstacle (bad), or (3) the vehicle reaches the end of the path (good). We can define the reward function as follows:
$$
r(s,a) = \begin{cases}
-1 & \text{if } p_y \notin [y_\text{min}, y_\text{max}] \\
-1 & \text{if } \exists o \in O \text{ such that } \sqrt{(p_x - o_x)^2 + (p_y - o_y)^2} \leq r_\text{min} \\
1 & \text{if } p_x \geq x_\text{max}
\end{cases}
$$
where $y_\text{min}$ and $y_\text{max}$ are the minimum and maximum y-coordinates of the path, $r_\text{min}$ is the minimum distance between the vehicle and an obstacle that constitutes a collision, and $x_\text{max}$ is the maximum x-coordinate of the path.
