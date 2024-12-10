import numpy as np
from scipy.linalg import expm  # For matrix exponential
import massParam as P

matrix_version = 2

A_og = np.array([[0.0, 1.0],
                      [-P.k / P.m, -P.b / P.m]])




def matrix_exponential_analytical(A, dt):
    #TODO: analytically write out the solution to e^(A*dt)
    
    return exp_matrix


def matrix_exponential(A):
    # TODO: Use the Cayley-Hamilton Simplification to symbolically compute e^(A*dt). 
    # The code is designed to simplify the experience for you by multiplying A by dt in the function call.
    # This will allow you to focus on simply computing e^(A).
    return exp_A


def compute_integral(u,x, A=A_og, dt=P.Ts):
    """
    Compute the integral:
        ∫ e^(A * (t-s)) * B * u(s) ds
    where u is a discrete-time array and A, B are constant matrices.
    This is accomplished using an Euler Approximation.

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
    if matrix_version == 1:
        exp_A_dt = matrix_exponential(A*dt)
    elif matrix_version == 2:
        exp_A_dt = matrix_exponential_analytical(A,dt)
    else:
        exp_A_dt = expm(A*dt)


    result = x + dt*exp_A_dt*u
    
    return result