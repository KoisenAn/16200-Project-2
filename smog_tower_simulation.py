import time

import math
import numpy as np
import matplotlib.pyplot as plt

#------------------- Parameters -------------------

GRAVITY = 9.81
EPSILON_0 = 8.854e-12

AIR_DENSITY = 1.225
AIR_VISCOSITY = 18.6e-6

AIR_FLOW_DIRECTION = np.array([0.0, -0.66])
TOWER_CHARGE_DENSITY = 1e-3

particle_density = 1200
particle_diameter = 5e-6
particle_charge = 1e-18

particle_mass = (math.pi/6) * particle_density * particle_diameter**3
particle_volume = np.pi/6 * particle_diameter**3
particle_area = np.pi * particle_diameter**2 / 4

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

        if np.all(Y[0::4] <= 0) or np.all(Y[1::4] <= 0):
            break
        
        if frame % (simulation_time / dt / 100) == 0:
            print(f"Time: {time.time()-start_time:.2f}s / Completion: {int(100 * t/t_end)}%")
        
        k1 = f(t, Y)
        k2 = f(t + dt/2, Y + dt/2 * k1)
        k3 = f(t + dt/2, Y + dt/2 * k2)
        k4 = f(t + dt, Y + dt * k3)
        
        Y += dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += dt
        
        t_values.append(t)
        Y_values.append(Y.copy())
    
    return np.array(t_values), np.array(Y_values)

#------------------- Charge Density Functions -------------------

'''
CHARGE_DENSITY_X_CONST = 1
CHARGE_DENSITY_Y_CONST = 3

DECAY_X = 0.3
DECAY_Y = 1

def find_charge_density(x, y):
    charge_density_x = CHARGE_DENSITY_X_CONST * np.exp(-(x) / DECAY_X)
    charge_density_y = CHARGE_DENSITY_Y_CONST * np.exp(-(7-y) / DECAY_Y)
    charge_density = charge_density_x + charge_density_y
    return charge_density
'''
    
CENTER_X = 0
CENTER_Y = 7
CONST_X = 4
CONST_Y = 0.01
PEAK_HEIGHT = 1
DECAY = 0.5

def find_charge_density(x, y):
    return PEAK_HEIGHT * np.exp(-(CONST_X * (x - CENTER_X)**2 + CONST_Y * (y - CENTER_Y)**2) / (2 * DECAY**2))

def electric_field_cloud(particle_x, particle_y, num_points=10):

    y_vals_lower = np.linspace(0, particle_y, num_points)
    y_vals_upper = np.linspace(particle_y, 7, num_points)

    rho_lower_y = np.array([find_charge_density(particle_x, y) for y in y_vals_lower])
    rho_upper_y = np.array([find_charge_density(particle_x, y) for y in y_vals_upper])
    
    integral_lower_y = np.trapz(rho_lower_y, y_vals_lower)
    integral_upper_y = np.trapz(rho_upper_y, y_vals_upper)

    electric_cloud_force_y = (particle_charge / (2 * EPSILON_0)) * (integral_lower_y - integral_upper_y)

    x_vals_lower = np.linspace(0, particle_x, num_points)
    x_vals_upper = np.linspace(particle_x, 1, num_points)
    
    rho_lower_x = np.array([find_charge_density(x, particle_y) for x in x_vals_lower])
    rho_upper_x = np.array([find_charge_density(x, particle_y) for x in x_vals_upper])
    
    integral_lower_x = np.trapz(rho_lower_x, x_vals_lower)
    integral_upper_x = np.trapz(rho_upper_x, x_vals_upper)

    electric_cloud_force_x = (particle_charge / (2 * EPSILON_0)) * (integral_lower_x - integral_upper_x)

    electric_cloud_force = np.array([electric_cloud_force_x, electric_cloud_force_y])

    return electric_cloud_force

#------------------- Simulation Function -------------------

def particle_simulation(t, Y):
    dYdt = np.zeros_like(Y)
    for i in range(N):
        
        index = i*4
        x, y, vx, vy = Y[index:index+4]

        # prevent moving at x = 0
        if x <= 0 or y <= 0:
            dYdt[index:index+4] = [0, 0, 0, 0]
            continue

        velocity = np.array([vx, vy])
        velocity_apparent = velocity - AIR_FLOW_DIRECTION
        velocity_magnitude = np.linalg.norm(velocity_apparent) + 1e-12
        
        gravity_force = np.array([0, - particle_mass * GRAVITY])

        buoyancy_force = np.array([0, AIR_DENSITY * particle_volume * GRAVITY])
        
        reynolds_number = max(AIR_DENSITY * particle_diameter * velocity_magnitude / AIR_VISCOSITY, 1e-12)
        Cd = 24 / reynolds_number
        drag_mag = 0.5 * AIR_DENSITY * Cd * particle_area * velocity_magnitude**2
        drag_force = -drag_mag * velocity_apparent / velocity_magnitude
        
        electric_plate_force = np.array([- particle_charge * TOWER_CHARGE_DENSITY / (2 * EPSILON_0), 0])
        
        electric_cloud_force = 

        total_force = gravity_force + buoyancy_force + drag_force + electric_plate_force
        acceleration = total_force / particle_mass
        
        dYdt[index:index+4] = [vx, vy, acceleration[0], acceleration[1]]

    return dYdt

#------------------- Run -------------------

Y0 = []
N = 20 

for i in range(N):
    x0 = (i+1) / (N+1)
    y0 = 7
    vx0 = 0
    vy0 = -0.66
    Y0.extend([x0, y0, vx0, vy0])

start_time = time.time()

''''
Y0 = [0.4, 7, 0, -0.66]
N = 1
'''

Y0 = np.array(Y0)

#------------------- Simulation Parameters -------------------

simulation_time = 0.01
dt = 0.0001

t_values, Y_values = runge_kutta_4(particle_simulation, Y0, (0, simulation_time), dt)

hit_particles = []

for i in range(N):
    x_vals = Y_values[:, i*4]
    hit_particles.append(np.any(x_vals <= 0))

print("----------- Results -----------")
print(f"Capture Rate: {(sum(hit_particles) / N) * 100}%")

#------------------- Plot Results -------------------

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

def animate_tragectory():
    fig, ax = plt.subplots(figsize=(6,8))
    ax.set_xlim(0,1)
    ax.set_ylim(0,7)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Particle Motion")

    downsample = 100

    lines = []
    x_vals_anim_all = []
    y_vals_anim_all = []

    for i in range(N):
        x_vals = Y_values[:, i*4][::downsample]
        y_vals = Y_values[:, i*4+1][::downsample]
        x_vals_anim_all.append(x_vals)
        y_vals_anim_all.append(y_vals)
        color = "green" if i in hit_particles else "red"
        line, = ax.plot([], [], lw=2, color=color)
        lines.append(line)

    max_frames = max(len(x) for x in x_vals_anim_all)

    def update(frame):
        for i, line in enumerate(lines):
            x_vals_anim = x_vals_anim_all[i]
            y_vals_anim = y_vals_anim_all[i]
            if frame < len(x_vals_anim):
                line.set_data(x_vals_anim[:frame+1], y_vals_anim[:frame+1])
        return lines

    ani = FuncAnimation(
        fig,
        update,
        frames=max_frames,
        interval=10,
        blit = True,
    )

    plt.show()

def plot_tragectory():
    fig2, ax2 = plt.subplots(figsize=(6,8))

    ax2.set_xlim(0,1)
    ax2.set_ylim(0,7)
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")
    ax2.set_title("Final Particle Trajectories")

    for i in range(N):
        x_vals = Y_values[:, i*4]
        y_vals = Y_values[:, i*4+1]

        color = "green" if i in hit_particles else "red"
        ax2.plot(x_vals, y_vals, color=color, lw=2)

    # draw tower wall
    ax2.axvline(0, color='black', linewidth=2)

    plt.show()

def plot_charge_density():
    fig3, ax3 = plt.subplots(figsize=(6,8))

    ax3.set_xlim(0,1)
    ax3.set_ylim(0,7)
    ax3.set_xlabel("x (m)")
    ax3.set_ylabel("y (m)")
    ax3.set_title("Charge Density")

    x = np.linspace(0.001, 1, 400)
    y = np.linspace(0.001, 7, 400)
    X, Y = np.meshgrid(x, y)
    Z = find_charge_density(X, Y)

    c = ax3.pcolormesh(X, Y, Z, shading='auto', cmap='viridis')
    fig3.colorbar(c, ax=ax3, label="Charge Density")

    plt.show()

plot_charge_density()
#animate_tragectory()
plot_tragectory()