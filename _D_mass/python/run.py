import matplotlib.pyplot as plt
import numpy as np
import massParam as P
from massDynamics import massDynamics
from ctrlStateFeedback import ctrlStateFeedback
from signalGenerator import signalGenerator
from massAnimation import massAnimation
from dataPlotter_exponential import dataPlotter
from scipy.linalg import expm  # For matrix exponential

use_self_defined_matrix = 2

A_og = np.array([[0.0, 1.0],
                      [-P.k / P.m, -P.b / P.m]])


import numpy as np


def matrix_exponential_analytical(A, dt):
    # Extract parameters from A assuming A is structured as specified
    b = -A[1, 1] * A[1, 0]
    k = -A[1, 0]
    m = 1 / A[1, 0]
    
    # Compute coefficients for the characteristic equation
    alpha = -b / (2 * m)
    omega_squared = (b**2 - 4 * k * m) / (4 * m**2)
    
    # Initialize the identity matrix
    I = np.eye(2)
    
    if omega_squared > 0:
        # Distinct real eigenvalues case
        omega = np.sqrt(omega_squared)
        exp_matrix = np.cosh(omega * dt) * I + (np.sinh(omega * dt) / omega) * (A - alpha * I)
    elif omega_squared == 0:
        # Repeated real roots case
        exp_matrix = I + dt * (A - alpha * I)
    else:
        # Complex conjugate roots case
        omega = np.sqrt(-omega_squared)
        exp_matrix = np.exp(alpha * dt) * (np.cos(omega * dt) * I + (np.sin(omega * dt) / omega) * (A - alpha * I))
    
    # Factor out e^(alpha * dt) for the complex and repeated root cases
    exp_matrix *= np.exp(alpha * dt)
    
    return exp_matrix


def matrix_exponential(A):
    # Step 1: Compute the Characteristic Polynomial:
    #  Find the characteristic polynomial χ(λ) of matrix A, which is λ² - tr(A)λ + det(A), 
    #  where tr(A) is the trace of A and det(A) is the determinant.
    #  So:
    #       Calculate trace and determinant
    trace_A = np.trace(A)
    det_A = np.linalg.det(A)
    
    # Step 2: Cayley-Hamilton Simplification: 
    #    Using the Cayley-Hamilton theorem, express A² in terms of A and the identity matrix I.
    #    Calculate A^2 using Cayley-Hamilton theorem
    #         A^2 - trace_A*A + det_A*I = 0  -->  A^2 = trace_A*A - det_A*I
    I = np.eye(2)  # Identity matrix
    A_squared = trace_A * A - det_A * I
    
    # Step 3: Matrix Exponential Series:
    #        Use the fact that A^n for n > 2 can be rewritten using a combination of A and I, and compute e^A as a finite sum.
    #        Compute matrix exponential using the series expansion up to the first few terms
    #           Again, we use the fact that A^2 is already simplified and all higher powers can be reduced
    exp_A = I  # Start with the identity matrix
    exp_A += A  # Add A
    exp_A += A_squared / 2  # Add A^2 / 2!
    
    # Calculate A^3 using A^2 and A
    A_cubed = np.dot(A, A_squared)
    exp_A += A_cubed / 6  # Add A^3 / 3!
    
    # Since A^2 is already in terms of A and I, A^4 and higher powers follow the same pattern and contribute less
    # We can limit the series to first few terms for practical purposes or continue as needed
    return exp_A


def compute_integral(u,x, A=A_og, dt=P.Ts):
    """
    Compute the integral:
        ∫ e^(A * (t-s)) * B * u(s) ds
    where u is a discrete-time array and A, B are constant matrices.

    Args:
        A (numpy.ndarray): Constant matrix A.
        B (numpy.ndarray): Constant matrix B.
        u (numpy.ndarray): Discrete-time input signal.
        dt (float): Time step between samples.

    Returns:
        numpy.ndarray: Array of results corresponding to each time step.
    """
   
    
    # # for t in range(n-1,n):#np.arange(P.t_start, t_cur +P.Ts, P.Ts):
    # integral_sum = np.zeros((A.shape[0], B.shape[1]))  # Temporary accumulator
    # for s in range(t + 1):  # Iterate over past values up to current time t
    #     e_term = expm(A * (t - s)*dt)  # Matrix exponential for (t-s)
    #     integral_sum += (e_term @ B * u[s])  # Add contribution of e^(A*(t-s)) * B * u(s)
    # result = integral_sum[0:2,0:]  # Multiply by dt and store result
    exp_A_dt = np.zeros_like(A)
    if use_self_defined_matrix == 1:
        exp_A_dt = matrix_exponential(A*dt)
    elif use_self_defined_matrix == 2:
        exp_A_dt = matrix_exponential_analytical(A,dt)
    else:
        exp_A_dt = expm(A*dt)


    result = x + dt*exp_A_dt*u
    
    return result

# instantiate satellite, controller, and reference classes
mass = massDynamics()
controller = ctrlStateFeedback()
reference = signalGenerator(amplitude=0.5, frequency=0.04)
disturbance = signalGenerator(amplitude=0.25)

# instantiate the simulation plots and animation
dataPlot = dataPlotter()
animation = massAnimation()
t = P.t_start  # time starts at t_start
y = mass.h()  # output of system at start of simulation
while t < P.t_end:  # main simulation loop
    # Propagate dynamics in between plot samples
    t_next_plot = t + P.t_plot
    while t < t_next_plot:  # updates control and dynamics at faster simulation rate
        r = reference.square(t)  # reference input
        d = disturbance.step(t)  # input disturbance
        n = 0.0  #noise.random(t)  # simulate sensor noise
        x = mass.state
        u = controller.update(r, x)  # update controller
        y = mass.update(u + d)  # propagate system
        t = t + P.Ts  # advance time by Ts
        x_p = compute_integral(u,x)
    # update animation and data plots
    animation.update(mass.state)
    dataPlot.update(t,  mass.state, x_p, u,r)
    plt.pause(0.0001)  # the pause causes the figure to be displayed during the simulation

# Keeps the program from closing until the user presses a button.
print('Press key to close')
plt.waitforbuttonpress()
plt.close()



