import math
import numpy as np
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt

GRAVITY = 9.81 # m/s^2
VACUUM_PERMITTIVITY = 8.854 * 10e-12 # C^2/(N•m^2)

AIR_DENSITY = 1.225 # kg/m^3 (25C)
AIR_VISCOSITY = 18.6 * 10e-6 # Pa•s (25C)
FORM_CONSTANT = 24

TOWER_CHARGE_DENSITY = 10e-4 # C

particle_density = 2.8 # kg / m^3
particle_diameter = 1 * 10e-6 # m
particle_charge = 1 * 10e-6 # C

particle_position = [0,0] 
particle_velocity = [0,0] # m/s
particle_acceleration = [0,0] # m/s^2

def system_of_equations(Y, t):

    x, y = Y
    
    velocity_magnitude = (particle_velocity[0] ** 2 + particle_velocity[1] ** 2) ** (0.5)

    force_gravity_y = -(math.pi / 6 * particle_density * GRAVITY * (particle_diameter ** 3))
    force_buoyancy_y = math.pi / 6 * AIR_DENSITY * GRAVITY * (particle_diameter ** 3)

    reynolds_number = AIR_DENSITY * velocity_magnitude / AIR_VISCOSITY
    form_drag = FORM_CONSTANT / reynolds_number

    drag_force_magnitude = 0.5 * AIR_DENSITY * form_drag * (math.pi / 4 * particle_density ** 2) * velocity_magnitude ** 2
    drag_force_x = - particle_velocity[0] / velocity_magnitude * drag_force_magnitude
    drag_force_y = - particle_velocity[1] / velocity_magnitude * drag_force_magnitude

    electric_field_plate_force_x = particle_charge * TOWER_CHARGE_DENSITY / (2 * VACUUM_PERMITTIVITY)

    #electric_field_cloud_force_x = 

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