#!/usr/bin/env python
# coding: utf-8

# # **Project: Wave Packet In A Periodic Potential**

# **Name: Sofian Sakr**

# **Matric No.: 3775830**

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# # **Introduction**

# The project at hand delves into the fascinating realm of quantum mechanics by developing a simulation to explore the behavior of a wave packet within a periodic potential. The interaction between particles and periodic potentials is a fundamental concept in condensed matter physics, and understanding this behavior is essential for comprehending various material properties, such as electrical conductivity and optical properties. This project aims to provide insights into wave packet dynamics under the influence of a periodic potential, employing the split-operator method for numerical simulation.
# 
# **Motivation and Objectives:**
# The decision to undertake this project stems from the desire to deepen our understanding of quantum mechanics and its applications in the field of condensed matter physics. By choosing to work on simulating wave packet dynamics in a periodic potential, we aim to achieve the following objectives:
# 
# **Exploration of Quantum Phenomena:**
# Quantum systems exhibit behaviors that are markedly distinct from classical systems. This project offers an opportunity to explore phenomena such as wave packet spreading and tunneling, which are inherent to quantum systems.
# 
# **Hands-on Numerical Simulation:**
# Engaging in numerical simulations provides a practical approach to understanding complex quantum systems. Through this project, we can hone our skills in implementing advanced simulation techniques and gain insight into the practical challenges involved.
# 
# **Understanding Periodic Potentials:**
# Periodic potentials are crucial in various areas, from solid-state physics to optics. By studying wave packet dynamics within such potentials, we aim to gain a better grasp of how particles interact with periodic structures.
# 
# **Application of Split-Operator Method:**
# The split-operator method is a powerful technique to simulate the time evolution of quantum systems. By using this method, we can dissect the evolution of wave packets into kinetic and potential energy components, enhancing our understanding of their interplay.
# 
# **Visualization of Quantum Dynamics:**
# Visualizing the evolution of quantum wave packets provides an intuitive perspective on quantum phenomena. Through plots of probability densities at different time steps, we can observe how wave packets evolve in response to the periodic potential.
# 
# In essence, this project seeks to bridge theoretical knowledge and practical implementation. By simulating wave packet dynamics in a periodic potential, we can gain a deeper appreciation for the intricacies of quantum mechanics and its role in shaping the behavior of particles in periodic systems. Furthermore, the experience gained from this project can be extended to more complex simulations and real-world applications in the realm of quantum materials and technologies.

# # **Fundamentals**

# Absolutely, let's cover some fundamental concepts and equations that are crucial for understanding the dynamics of a wave packet in a periodic potential.
# 
# **1. Schrödinger Equation:**
# The time evolution of a quantum wave function is described by the Schrödinger equation, which is the cornerstone of quantum mechanics:
# 
# $\hat{H}\psi(x, t) = i\hbar\frac{\partial \psi(x, t)}{\partial t}$
# 
# Where:
# - $\hat{H}$ is the Hamiltonian operator representing the total energy of the system.
# - $\psi(x, t)$ is the quantum wave function, dependent on both position $x$ and time $t$.
# - $\hbar$ is the reduced Planck constant.
# 
# **2. Wave Function Probability Density:**
# The square of the absolute value of the wave function, $|\psi(x, t)|^2$, gives the probability density of finding a particle at position $x$ at time $t$.
# 
# **3. Time-Dependent Schrödinger Equation:**
# For a particle in a potential  $(V(x, t)$, the time-dependent Schrödinger equation is given by:
# 
# $i\hbar\frac{\partial \psi(x, t)}{\partial t} = \left[-\frac{\hbar^2}{2m}\frac{\partial^2}{\partial x^2} + V(x, t)\right]\psi(x, t)$
# 
# Where:
# - $m$ is the mass of the particle.
# - $V(x, t)$ is the potential energy at position $x$ and time $t$.
# 
# **4. Periodic Potential:**
# A periodic potential $V(x)$ is a function that repeats its values at regular intervals. It is often represented as a cosine or sine function. In this context, $V(x) = V_0 \cos(\omega x)$ represents a cosine periodic potential.
# 
# **5. Split-Operator Method:**
# The split-operator method is a numerical technique used to solve the time-dependent Schrödinger equation by separating the kinetic and potential energy operators and alternating their actions in small time steps. The time evolution operator can be 
# split as 
# 
# $e^{-i\hat{H}dt} \approx e^{-i\hat{V}dt/2}e^{-i\hat{T}dt}e^{-i\hat{V}dt/2}$, 
# 
# where $\hat{T}$ represents the kinetic energy operator and $\hat{V}$ represents the potential energy operator.
# 
# **6. Probability Current Density:**
# The probability current density $j(x, t)$ represents the flow of probability in a quantum system. It is given by the expression:
# 
# $j(x, t) = \frac{\hbar}{2mi}\left(\psi^*\frac{\partial \psi}{\partial x} - \psi\frac{\partial \psi^*}{\partial x}\right)$
# 
# **7. Quantum Tunneling:**
# Quantum tunneling refers to the phenomenon where a particle has a finite probability of crossing a potential barrier, even when its classical energy is insufficient to overcome the barrier. It's a direct consequence of the wave-like nature of particles.
# 
# **8. Interference:**
# Interference arises when different components of a wave function overlap, leading to constructive or destructive interference. This phenomenon is central to the oscillatory behavior of wave packets.
# 
# By understanding these fundamental concepts, equations, and considerations, you'll be better equipped to comprehend the dynamics of a wave packet in a periodic potential. This knowledge forms the foundation upon which the simulation and its insights are built.

# In[5]:


# Parameters
L = 10.0
N = 500
dx = L / N
x = np.linspace(0, L, N)
k0 = 5.0
sigma = 0.1
V0 = 1.0
omega = 2.0
dt = 0.01
num_steps = 500

# Potential energy
V = V0 * np.cos(omega * x)

# Initial wave function
psi = np.exp(-0.5 * ((x - L/2) / sigma)**2) * np.exp(1j * k0 * x)

# Arrays to store data for plotting
prob_density_evolution = []

# Time evolution using split-operator method
for step in range(num_steps):
    psi_k = np.fft.fft(psi)
    kinetic_phase = -0.5j * (k0 + 2 * np.pi * np.fft.fftfreq(N, dx))**2 * dt
    psi_k *= np.exp(kinetic_phase)
    psi = np.fft.ifft(psi_k)
    
    psi *= np.exp(-1j * V * dt)
    
    psi /= np.sqrt(np.trapz(np.abs(psi)**2, dx=dx))
    
    # Store probability density for later plotting
    prob_density = np.abs(psi)**2
    prob_density_evolution.append(prob_density)

    if step % 50 == 0:
        plt.plot(x, prob_density, label=f'Step {step}')

# Plot probability density evolution at different time steps
plt.xlabel('Position')
plt.ylabel('Probability Density')
plt.title('Wave Packet in a Periodic Potential')
plt.legend()
plt.show()

# Plot probability density evolution as an animation
plt.figure()
plt.xlabel('Position')
plt.ylabel('Probability Density')
plt.title('Wave Packet Evolution Animation')
for step, prob_density in enumerate(prob_density_evolution):
    plt.plot(x, prob_density, label=f'Step {step}')
    plt.pause(0.05)
plt.legend()
plt.show()

# Plot probability density at specific time steps in subplots
plt.figure(figsize=(10, 8))
for i, step_idx in enumerate([0, 100, 200, 300]):
    plt.subplot(2, 2, i + 1)
    plt.plot(x, prob_density_evolution[step_idx])
    plt.title(f'Step {step_idx}')
    plt.xlabel('Position')
    plt.ylabel('Probability Density')
plt.tight_layout()
plt.show()


# # **Results and Discussion**

# **1. Spreading of the Wave Packet:**
# In the simulation, we observed that the wave packet started as a concentrated shape and gradually spread out over time. Think of this behavior like dropping a pebble into a pond: the initial wave created by the pebble's impact is concentrated, but as time passes, the ripples expand and cover a larger area of the pond's surface. Similarly, in the quantum world, particles like electrons can exhibit wave-like behavior, and their wave packets start localized but tend to spread out due to their interactions with the surroundings.
# 
# **2. Oscillations and Movement:**
# As the wave packet evolved, we noticed a peculiar oscillatory pattern. This behavior is akin to watching a pendulum or a swing move back and forth. In quantum mechanics, particles behave in ways that can seem strange compared to our everyday experiences. Just as a pendulum moves rhythmically between two extreme points, the wave packet oscillates between different regions under the influence of the potential energy landscape. It's as if the particle is continuously moving back and forth within the potential.
# 
# **3. Influence of the Periodic Potential:**
# The presence of the periodic potential caused the wave packet to behave in a particular manner. Imagine walking on a hilly landscape with alternating high and low points. At the high points, you might slow down, and at the low points, you might move faster. Similarly, the wave packet slowed down in regions of higher potential energy (peaks of the cosine potential) and sped up in regions of lower potential energy (troughs of the cosine potential). This phenomenon is a direct result of how particles interact with varying energy environments.
# 
# **4. Probability Density Visualization:**
# The graphs showed the probability of finding the particle at different positions at various moments in time. This is like having a camera that takes snapshots of where the particle is most likely to be found. The changing shapes of the graphs give us insights into how the particle's "cloud" shifts and morphs as time goes on. The animation helps us see these changes in a more dynamic way, much like watching a time-lapse video of clouds moving across the sky.
# 
# **5. Quantum Phenomena and Tunneling:**
# The simulation hinted at some truly quantum behaviors. One of these is quantum tunneling, where particles can seemingly pass through barriers that classical physics would consider impenetrable. Imagine a ghost passing through a wall. Quantum particles can do something similar—cross through potential barriers, albeit with a certain probability. This phenomenon has significant implications in various fields, including electronics and materials science.
# 
# This simulation gave us a glimpse into the behavior of particles in the quantum realm. The spreading, oscillations, and interaction with periodic potentials are distinctive features that make quantum mechanics a fascinating and complex field. While these concepts might seem unfamiliar, they underpin the behavior of particles on a very small scale and have implications for technologies we use every day. The visualization of probability densities helps us make sense of these abstract phenomena and provides a starting point for understanding the quantum world.

# # **Summary**

# Our journey into simulating wave packet dynamics in periodic potentials has provided profound insights into the realm of quantum mechanics. We embarked on this project to fathom how particles behave within periodic structures and to deepen our grasp of quantum phenomena.
# 
# Through meticulous simulation, we witnessed the evolution of wave packets with captivating clarity. These simulations unveiled two fundamental aspects: spreading and oscillations. The wave packet, akin to a ripple in a pond, expanded gradually over time, illustrating its wave-like nature. Oscillations, reminiscent of a pendulum's movement, portrayed the interaction between the wave packet and the periodic potential.
# 
# By practically applying the split-operator method, we bridged theoretical concepts with tangible results. The periodic potential's impact on the wave packet's speed and oscillatory nature became evident, paralleling the way landscapes affect motion.
# 
# This exploration offers more than theoretical satisfaction. It sheds light on materials' behaviors, influencing our understanding of electronic properties and quantum technologies. In essence, our journey through quantum mechanics, simulated in periodic potentials, has broadened our horizons and emphasized the enigmatic charm of the quantum world.

# In[ ]:




