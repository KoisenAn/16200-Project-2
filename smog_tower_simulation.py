import math
import numpy as np
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt

AIR_DENSITY = 1.225 # kg/m^3 (25C)
AIR_VISCOSITY = 18.6 * 10e-6 # Pa•s (25C)
GRAVITY = 9.81 # m/s^2

CHARGE_OF_TOWER = 3 # Coulombs

particle_density = 1 * 10e-6 # m
particle_diameter = 1 * 10e-6 # m

particle_position = [0,0] 
particle_velocity = [0,0] # m/s
particle_acceleration = [0,0] # m/s^2

def system_of_equations(Y, t):

    x, y = Y
    
    force_gravity_y = math.pi / 6 * particle_density * GRAVITY * (particle_diameter ** 3)
    force_buoyancy_y = math.pi / 6 * AIR_DENSITY * GRAVITY * (particle_diameter ** 3)

    drag_force_magnitude = 0.5 * 

    dxdt = x
    dxdt = -2 * x - y
    
    return [dxdt, dxdt]

y0 = [1, 0]
t_span = np.arange(0, 10, 0.01)

solution = solve_ivp(system_of_equations, y0, t_span)

x1_values = solution[:, 0]
x2_values = solution[:, 1]

plt.figure(figsize=(10, 6))
plt.plot(t_span, x1_values, label='$x_1(t)$')
plt.plot(t_span, x2_values, label='$x_2(t)$')
plt.xlabel('Time (t)')
plt.ylabel('Variables')
plt.title('Solution of a System of Two ODEs in Python')
plt.legend()
plt.grid(True)
plt.show()