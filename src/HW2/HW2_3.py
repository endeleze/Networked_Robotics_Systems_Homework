import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial
from tqdm import tqdm

class ViscekModel:
    def __init__(self, num_particles, box_size=50.0, noise_strength=0.13, num_time_steps=300, initial_speed=1, neighbor_radius=1, time_step_size=1):
        self.num_particles = num_particles
        self.box_size = box_size
        self.noise_strength = noise_strength
        self.num_time_steps = num_time_steps
        self.initial_speed = initial_speed
        self.neighbor_radius = neighbor_radius
        self.time_step_size = time_step_size
        self.x, self.y, self.orientation_x, self.orientation_y, self.theta = self.run_simulation()
        
    def initialize_particles(self):
        x0 = np.random.rand(self.num_particles) * self.box_size
        y0 = np.random.rand(self.num_particles) * self.box_size
        angle0 = np.random.uniform(-np.pi, np.pi, size=self.num_particles)
        return x0, y0, angle0

    def initialize_arrays(self):
        x = np.zeros((self.num_time_steps, self.num_particles), dtype=float)
        y = np.zeros((self.num_time_steps, self.num_particles), dtype=float)
        theta = np.zeros((self.num_time_steps, self.num_particles), dtype=float)
        orientation_x = np.zeros((self.num_time_steps, self.num_particles), dtype=float)
        orientation_y = np.zeros((self.num_time_steps, self.num_particles), dtype=float)
        return x, y, theta, orientation_x, orientation_y

    def update_positions(self, positions, theta):
        positions[:, 0] += self.initial_speed * np.cos(theta) * self.time_step_size
        positions[:, 1] += self.initial_speed * np.sin(theta) * self.time_step_size
        positions[positions > self.box_size] -= self.box_size
        positions[positions < 0] += self.box_size
        return positions

    def run_simulation(self):
        x0, y0, angle0 = self.initialize_particles()
        x, y, theta, orientation_x, orientation_y = self.initialize_arrays()

        x[0, :] = x0
        y[0, :] = y0
        theta[0, :] = angle0
        orientation_x[0, :] = np.cos(angle0)
        orientation_y[0, :] = np.sin(angle0)

        positions = np.zeros((self.num_particles, 2))
        positions[:, 0] = x[0, :]
        positions[:, 1] = y[0, :]

        for t in tqdm(range(self.num_time_steps - 1),desc=f'Simulation of {self.num_particles} particles'):
            noise = self.noise_strength * np.random.uniform(-np.pi, np.pi, size=self.num_particles)

            for n in range(self.num_particles):
                distance_matrix = scipy.spatial.distance.pdist(positions)
                distance_matrix = scipy.spatial.distance.squareform(distance_matrix)
                neighbors = [nn for nn in range(self.num_particles) if n != nn if distance_matrix[n, nn] <= self.neighbor_radius]

                if len(neighbors) > 0:
                    neighbor_orient = [np.exp(theta[t, neighbor]*1j) for neighbor in neighbors]
                    average_velocity = sum(neighbor_orient) / len(neighbors)
                    theta[t + 1, n] = np.angle(average_velocity) + noise[n]
                else:
                    theta[t + 1, n] = theta[t, n]

            orientation_x[t + 1, :] = np.cos(theta[t + 1, :])
            orientation_y[t + 1, :] = np.sin(theta[t + 1, :])

            positions = self.update_positions(positions, theta[t + 1, :])

            x[t + 1, :] = positions[:, 0]
            y[t + 1, :] = positions[:, 1]

        return x, y, orientation_x, orientation_y, theta

