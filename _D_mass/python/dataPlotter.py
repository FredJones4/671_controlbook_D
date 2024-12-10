import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import massParam as P
from scipy.linalg import expm  # For matrix exponential
plt.ion()  # enable interactive drawing

A_og = np.array([[0.0, 1.0],
                      [-P.k / P.m, -P.b / P.m]])
B_og = np.array([[0.0],
                      [1.0 / P.m]])

def f(self, state, u):
    #  Return xdot = f(x,u),
    z = state.item(0)
    zdot = state.item(1)
    force = u
    # The equations of motion.
    zddot = (force - self.b*zdot - self.k*z)/self.m
    # build xdot and return
    xdot = np.array([[zdot], [zddot]])
    return xdot

def rk4_step(self, x, u):
    # Integrate ODE using Runge-Kutta RK4 algorithm
    F1 = f(x, u)
    F2 = f(x + P.Ts / 2 * F1, u)
    F3 = f(x + P.Ts / 2 * F2, u)
    F4 = f(x + P.Ts * F3, u)
    x += P.Ts / 6 * (F1 + 2*F2 + 2*F3 + F4)
    return x


def compute_integral(u,x, t_cur, A=A_og, B=B_og, dt=P.Ts):
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
    n = len(u)
    t = n - 1
   
    
    # # for t in range(n-1,n):#np.arange(P.t_start, t_cur +P.Ts, P.Ts):
    # integral_sum = np.zeros((A.shape[0], B.shape[1]))  # Temporary accumulator
    # for s in range(t + 1):  # Iterate over past values up to current time t
    #     e_term = expm(A * (t - s)*dt)  # Matrix exponential for (t-s)
    #     integral_sum += (e_term @ B * u[s])  # Add contribution of e^(A*(t-s)) * B * u(s)
    # result = integral_sum[0:2,0:]  # Multiply by dt and store result

    result = x + dt*expm(A*dt)*u[t]
    
    return result





class dataPlotter:
    def __init__(self):
        # Number of subplots = num_of_rows*num_of_cols
        self.num_rows = 2    # Number of subplot rows
        self.num_cols = 1    # Number of subplot columns
        # Crete figure and axes handles
        self.fig, self.ax = plt.subplots(
            self.num_rows, self.num_cols, sharex=True)
        # Instantiate lists to hold the time and data histories
        self.time_history = []  # time
        self.z_ref_history = []  # reference position
        self.z_history = []  # position z
        self.force_history = []  # control force

        ###############################
        self.zpred_history = []

        ################################


        # create a handle for every subplot.
        self.handle = []
        self.handle.append(
            myPlot(self.ax[0], ylabel='z(m)', title='Mass Data'))
        self.handle.append(
            myPlot(self.ax[1], xlabel='t(s)', ylabel='force(N)'))

    def update(self, t: float, states: np.ndarray, states_pred: np.ndarray, ctrl: float, reference: float = 0.):
        '''
            Add to the time and data histories, and update the plots.
            state order is assumed to be [z, z_dot]
        '''
        # update the time history of all plot variables
        self.time_history.append(t)  # time
        self.z_ref_history.append(reference)  # reference mass position
        self.z_history.append(states.item(0))  # mass position
        self.force_history.append(ctrl)  # force on the base


        #######################################
        # states_pred = #compute_integral(u=self.force_history, x=states, t_cur=t)
        self.zpred_history.append(states_pred.item(0))

        #######################################

        # update the plots with associated histories
        self.handle[0].update(self.time_history, [
                              self.z_history, self.z_ref_history,
                              self.zpred_history
                              ])
        self.handle[1].update(self.time_history, [self.force_history])


class myPlot:
    ''' 
        Create each individual subplot.
    '''

    def __init__(self, ax,
                 xlabel='',
                 ylabel='',
                 title='',
                 legend=None):
        ''' 
            ax - This is a handle to the  axes of the figure
            xlable - Label of the x-axis
            ylable - Label of the y-axis
            title - Plot title
            legend - A tuple of strings that identify the data. 
                     EX: ("data1","data2", ... , "dataN")
        '''
        self.legend = legend
        self.ax = ax                  # Axes handle
        self.colors = ['b', 'g', 'r', 'c', 'm', 'y', 'b']
        # A list of colors. The first color in the list corresponds
        # to the first line object, etc.
        # 'b' - blue, 'g' - green, 'r' - red, 'c' - cyan, 'm' - magenta
        # 'y' - yellow, 'k' - black
        self.line_styles = ['-', '-', '--', '-.', ':']
        # A list of line styles.  The first line style in the list
        # corresponds to the first line object.
        # '-' solid, '--' dashed, '-.' dash_dot, ':' dotted
        self.line = []
        # Configure the axes
        self.ax.set_ylabel(ylabel)
        self.ax.set_xlabel(xlabel)
        self.ax.set_title(title)
        self.ax.grid(True)
        # Keeps track of initialization
        self.init = True

    def update(self, time, data):
        ''' 
            Adds data to the plot.  
            time is a list, 
            data is a list of lists, each list corresponding to a line on the plot
        '''
        if self.init == True:  # Initialize the plot the first time routine is called
            for i in range(len(data)):
                # Instantiate line object and add it to the axes
                self.line.append(Line2D(time,
                                        data[i],
                                        color=self.colors[np.mod(
                                            i, len(self.colors) - 1)],
                                        ls=self.line_styles[np.mod(
                                            i, len(self.line_styles) - 1)],
                                        label=self.legend if self.legend != None else None))
                self.ax.add_line(self.line[i])
            self.init = False
            # add legend if one is specified
            if self.legend != None:
                plt.legend(handles=self.line)
        else:  # Add new data to the plot
            # Updates the x and y data of each line.
            for i in range(len(self.line)):
                self.line[i].set_xdata(time)
                self.line[i].set_ydata(data[i])
        # Adjusts the axis to fit all of the data
        self.ax.relim()
        self.ax.autoscale()
