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
TOWER_CHARGE_DENSITY = 5e-6

# PM2.5
particle_density = 1200
particle_diameter = 5e-6
particle_charge = 1e-16

# PM10
'''
particle_density = 1000
particle_diameter = 2e-5
particle_charge = 2.9e-15
'''

particle_mass = (math.pi/6) * particle_density * particle_diameter**3
particle_volume = np.pi/6 * particle_diameter**3
particle_area = np.pi * particle_diameter**2 / 4

#------------------- RK4 Solver -------------------

def runge_kutta_4(f, Y0, dt, t_end=None, stop_when_all_done=True, output=True):

    t = 0
    Y = Y0.copy()

    n_particles = len(Y0) // 4
    active = np.ones(n_particles, dtype=bool)

    t_values = [t]
    Y_values = [Y.copy()]

    frame = 0
    last_percent = -1
    start_time = time.time()

    while True:
        frame += 1

        x = Y[0::4]
        y = Y[1::4]

        active &= (x > 0) & (y > 0)

        finished = np.sum(~active)

        if stop_when_all_done and not np.any(active):
            print(f"Frame: {frame} | Finished: {finished}/{n_particles} | Time: {time.time()-start_time:.2f}s")
            break

        if t_end is not None and t >= t_end:
            break

        if t_end is None:
            if frame % 1000 == 0 and output:
                print(f"Frame: {frame} | Finished: {finished}/{n_particles} | Time: {time.time()-start_time:.2f}s")
        else:
            percent = int(100 * (t / t_end))

            if percent != last_percent:
                last_percent = percent
                print(f"Progress: {percent}% | Finished: {finished}/{n_particles} | Time: {time.time()-start_time:.2f}s | Sim Time: {t:.2f}s")

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

from scipy.special import erf

CENTER_X = 0
CENTER_Y = 7
CONST_X = 4
CONST_Y = 0.1
DECAY = 0.8

PEAK_HEIGHT = 1e-8

def find_charge_density(x, y):
    return PEAK_HEIGHT * np.exp(-(CONST_X * (x - CENTER_X)**2 + CONST_Y * (y - CENTER_Y)**2) / (2 * DECAY**2))

A_x = CONST_X / (2 * DECAY**2)
A_y = CONST_Y / (2 * DECAY**2)

def gaussian_integral(a, b, center, A):
    return np.sqrt(np.pi) / (2 * np.sqrt(A)) * (
        erf(np.sqrt(A) * (b - center)) -
        erf(np.sqrt(A) * (a - center))
    )

def find_electric_cloud_force(particle_x, particle_y):

    gx = np.exp(-A_x * (particle_x - CENTER_X)**2)
    gy = np.exp(-A_y * (particle_y - CENTER_Y)**2)

    lower_y = gaussian_integral(0, particle_y, CENTER_Y, A_y)
    upper_y = gaussian_integral(particle_y, 7, CENTER_Y, A_y)

    integral_lower_y = PEAK_HEIGHT * gx * lower_y
    integral_upper_y = PEAK_HEIGHT * gx * upper_y

    electric_cloud_force_y = (particle_charge / (2 * EPSILON_0)) * (
        integral_lower_y - integral_upper_y
    )

    lower_x = gaussian_integral(0, particle_x, CENTER_X, A_x)
    upper_x = gaussian_integral(particle_x, 1, CENTER_X, A_x)

    integral_lower_x = PEAK_HEIGHT * gy * lower_x
    integral_upper_x = PEAK_HEIGHT * gy * upper_x

    electric_cloud_force_x = (particle_charge / (2 * EPSILON_0)) * (
        integral_lower_x - integral_upper_x
    )

    return np.array([electric_cloud_force_x, electric_cloud_force_y])

#------------------- Simulation Function -------------------

def particle_simulation(t, Y, output=True):
    global last_print_step
    dYdt = np.zeros_like(Y)
    step = int(t / dt)
    for i in range(N):
        
        index = i*4
        x, y, vx, vy = Y[index:index+4]

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
        
        electric_cloud_force = find_electric_cloud_force(x,y)
        #electric_cloud_force = 0

        total_force = gravity_force + buoyancy_force + drag_force + electric_plate_force + electric_cloud_force
        acceleration = total_force / particle_mass
        
        if output and i == 0 and step % 1000 == 0 and step != last_print_step:
            if step % 10000 == 0:
                last_print_step = step
                print(f"-------------- Particle {i+1} --------------")
                print(f"{'Gravity':<16} Fx={gravity_force[0]:>12.3e}  Fy={gravity_force[1]:>12.3e}")
                print(f"{'Buoyancy force':<16} Fx={buoyancy_force[0]:>12.3e}  Fy={buoyancy_force[1]:>12.3e}")
                print(f"{'Drag force':<16} Fx={drag_force[0]:>12.3e}  Fy={drag_force[1]:>12.3e}")
                print(f"{'Plate force':<16} Fx={electric_plate_force[0]:>12.3e}  Fy={electric_plate_force[1]:>12.3e}")
                print(f"{'Cloud force':<16} Fx={electric_cloud_force[0]:>12.3e}  Fy={electric_cloud_force[1]:>12.3e}")
                print(f"{'Velocity':<16} vx={vx:>12.3e}  vy={vy:>12.3e}")
                print(f"{'Acceleration':<16} ax={acceleration[0]:>12.3e}  ay={acceleration[1]:>12.3e}")

            force_history["gravity"].append(gravity_force.copy())
            force_history["buoyancy"].append(buoyancy_force.copy())
            force_history["drag"].append(drag_force.copy())
            force_history["plate"].append(electric_plate_force.copy())
            force_history["cloud"].append(electric_cloud_force.copy())
            force_history["velocity"].append(np.array([vx, vy]))
            force_history["acceleration"].append(acceleration.copy())
            force_history["position"].append(np.array([x, y]))
            force_history["time"].append(t)

        dYdt[index:index+4] = [vx, vy, acceleration[0], acceleration[1]]

    return dYdt

#------------------- Run -------------------

Y0 = []
N = 4

for i in range(N):
    x0 = (i+1) / (N+1)
    y0 = 7
    vx0 = 0
    vy0 = -0.66
    Y0.extend([x0, y0, vx0, vy0])

start_time = time.time()

Y0 = np.array(Y0)

last_print_step = -1
simulation_time = 1
dt = 0.0001

hit_particles = []
force_history = {
    "gravity": [],
    "buoyancy": [],
    "drag": [],
    "plate": [],
    "cloud": [],
    "velocity": [],
    "acceleration": [],
    "position": [],
    "time": []
}

t_values, Y_values = runge_kutta_4(particle_simulation, Y0, dt, stop_when_all_done=True)

for i in range(N):
    x_vals = Y_values[:, i*4]
    hit_particles.append(np.any(x_vals <= 0))

for key in force_history:
    force_history[key] = np.array(force_history[key])

print("----------- Results -----------")
print(f"Time taken: {time.time() - start_time:.2f}s")
print(f"Capture Rate: {(sum(hit_particles) / N) * 100:.2f}%")

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
        color = "green" if hit_particles[i] else "red"
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
    ax2.set_title("PM2.5 Particle Trajectories")

    for i in range(N):
        x_vals = Y_values[:, i*4]
        y_vals = Y_values[:, i*4+1]

        color = "green" if hit_particles[i] else "red"
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
    fig3.colorbar(c, ax=ax3, label=r"Charge Density $\rho$ (C/m$^3$)")

    plt.show()

def plot_forces_over_time(skip_time=0.25):
    t = np.array(force_history["time"])

    mask = t > skip_time
    
    plt.plot(t[mask], np.linalg.norm(force_history["gravity"][mask], axis=1), label="Gravity Force Magnitude")
    plt.plot(t[mask], np.linalg.norm(force_history["drag"][mask], axis=1), label="Drag Force Magnitude")
    plt.plot(t[mask], np.linalg.norm(force_history["cloud"][mask], axis=1), label="Cloud Force Magnitude")
    plt.plot(t[mask], np.linalg.norm(force_history["plate"][mask], axis=1), label="Plate Force Magnitude")
    plt.plot(t[mask], np.linalg.norm(force_history["buoyancy"][mask], axis=1), label="Buoyancy Force Magnitude")
    
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.yscale("log")
    plt.legend()
    plt.show()

plot_charge_density()
plot_forces_over_time()
animate_tragectory()
plot_tragectory()