import time

import math
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

#------------------- Parameters -------------------

GRAVITY = 9.81
VACUUM_PERMITTIVITY = 8.854e-12

AIR_DENSITY = 1.225
AIR_VISCOSITY = 18.6e-6

AIR_FLOW_DIRECTION = np.array([0.0, -0.66])
TOWER_CHARGE_DENSITY = 1e-4

particle_density = 1200
particle_diameter = 5e-6
particle_charge = 1e-18

particle_mass = (math.pi/6) * particle_density * particle_diameter**3
particle_volume = np.pi/6 * particle_diameter**3
particle_area =  np.pi * particle_diameter**2 / 4

#------------------- Simulation Function -------------------
frame = 0
def particle_simulation(t, Y):
    global frame
    frame += 1
    #print(f"Time: {time.time()-start_time:.2f} (s) / Frame: {frame} / Simulation Time: t={t:.5f}")
    if (frame % 1000 == 0):
        print(f"Time: {time.time()-start_time:.2f} (s) / Frame: {frame} / Completion: {100 * t/simulation_time:.2f}%")
    dYdt = np.zeros_like(Y)
    for i in range(N):
        
        index = i*4
        x, y, vx, vy = Y[index:index+4]

        # prevent moving at x = 0
        if x <= 0:
            dYdt[index:index+4] = [0, 0, 0, 0]
            continue

        velocity = np.array([vx, vy])
        v_app = velocity - AIR_FLOW_DIRECTION
        v_mag = np.linalg.norm(v_app) + 1e-12
        
        # gravity
        gravity_force = np.array([0, - particle_mass * GRAVITY])
        
        # buoyancy
        buoyancy_force = np.array([0, AIR_DENSITY * particle_volume * GRAVITY])
        
        # drag
        reynolds_number = max(AIR_DENSITY * particle_diameter * v_mag / AIR_VISCOSITY, 1e-12)
        Cd = 24 / reynolds_number
        drag_mag = 0.5 * AIR_DENSITY * Cd * particle_area * v_mag**2
        drag_force = -drag_mag * v_app / v_mag
        
        # electric plate_force
        electric_plate_force = np.array([- particle_charge * TOWER_CHARGE_DENSITY / (2 * VACUUM_PERMITTIVITY), 0])
        
        # electric cloud force
        #electric_cloud_force = 

        total_force = gravity_force + buoyancy_force + drag_force + electric_plate_force
        acceleration = total_force / particle_mass
        
        # prevent going below ground
        if y < 0:
            y = 0
            if vy < 0:
                vy = 0
        
        dYdt[index:index+4] = [vx, vy, acceleration[0], acceleration[1]]

    return dYdt

#------------------- Run -------------------

Y0 = []
N = 0
for i in range(1, 15):
    for j in range(1, 11):
        x0 = j * 0.1
        y0 = i / 2
        vx0 = 0
        vy0 = -0.66
        Y0.extend([x0, y0, vx0, vy0])
        N += 1

start_time = time.time()

'''
Y0 = [0.5, 7, 0, -0.66]
N = 1

Y0 = np.array(Y0)

simulation_time = 20 # seconds

t_span = (0, simulation_time)
t_eval = np.linspace(0, simulation_time, 2000)

start_time = time.time()
solution = solve_ivp(
    particle_simulation,
    t_span,
    Y0,
    t_eval=t_eval,
    rtol=1e-3,
    method='Radau',
)
print(f"Time Taken: {time.time()-start_time:.2f}")

x_vals = solution.y[0]
y_vals = solution.y[1]

#------------------- Plotting -------------------

num_plot_points = 100
plt.figure(figsize=(8,6))
for i in range(N):
    x_vals = solution.y[i*4]
    y_vals = solution.y[i*4+1]

    indexs = np.linspace(0, len(x_vals)-1, num_plot_points, dtype=int)
    x_plot = x_vals[indexs]
    y_plot = y_vals[indexs]

    plt.plot(x_plot, y_plot)

plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Trajectories of multiple particles")
plt.xlim(0,1)
plt.ylim(0,7)
plt.grid(True)
plt.show()
'''

#------------------- RK4 Solver -------------------
def runge_kutta_4(f, Y0, t_span, dt):
    t0, t_end = t_span
    t_values = [t0]
    Y_values = [Y0.copy()]
    
    Y = Y0.copy()
    t = t0
    frame = 0
    
    start_time = time.time()
    while t < t_end:
        frame += 1
        
        if frame % 1000 == 0:
            print(f"Time: {time.time()-start_time:.2f}s / Frame: {frame} / Completion: {100 * t/t_end:.2f}%")
        
        k1 = f(t, Y)
        k2 = f(t + dt/2, Y + dt/2 * k1)
        k3 = f(t + dt/2, Y + dt/2 * k2)
        k4 = f(t + dt, Y + dt * k3)
        
        Y += dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += dt
        
        t_values.append(t)
        Y_values.append(Y.copy())
    
    return np.array(t_values), np.array(Y_values)

#------------------- Simulation Parameters -------------------
simulation_time = 20
dt = 0.0001

#------------------- Run RK4 -------------------
t_values, Y_values = runge_kutta_4(particle_simulation, Y0, (0, simulation_time), dt)

#------------------- Plot Results -------------------
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
num_plot_points = 2000
for i in range(N):
    x_vals = Y_values[:, i*4]
    y_vals = Y_values[:, i*4+1]
    
    indexs = np.linspace(0, len(x_vals)-1, num_plot_points, dtype=int)
    plt.plot(x_vals[indexs], y_vals[indexs])

print(y_vals)

plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Particle Trajectories")
plt.xlim(0, 1)
plt.ylim(0, 7)
plt.grid(True)
plt.show()