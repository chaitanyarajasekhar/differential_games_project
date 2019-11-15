import numpy as np
from scipy.integrate import odeint
# import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plot1

class Plant2Player:
    def __init__(self, initial_state, dt, num_players=2, params=None):
        '''
            intializing the equations
        '''
        # self.state        = np.array([3,-1])
        self.state        = initial_state
        self.dt           = dt # change it take the value from params
        self.time         = []
        self.state_traj   = []
        self.current_time = 0.0
        self.state_traj.append(self.state.copy())
        self.time.append(0.)
        #
        self.num_players = num_players

    def g1(x):
        pass

    def g2(x):
        pass

    def nextState(self,input_u):
        '''
        '''
        # NOTE: change the input in the args
        state_dot  = self.stateEquation(self.state,0,input_u)
        self.state = state_dot * self.dt + self.state
        # next_state = odeint(Plant2Player.stateEquation2P,self.state,np.array([0.0, self.dt]),args=(input,))

        # self.state        = next_state[-1,:]
        self.current_time = self.current_time + self.dt

        # self.state_traj.append(next_state[-1,:])
        self.state_traj.append(self.state)

        self.time.append(self.current_time)

        return self.state

    def stateEquation2P(state,t,input_u):
        '''
            equations 68 and 69
        '''
        # current state
        x1   = state[0]
        x2   = state[1]
        u1   = input_u[0]
        u2   = input_u[1]
        # state equations
        f_x   = np.array([x2 - 2*x1, \
                -0.5*x1-x2+0.25*x2*(np.cos(2*x1)+2)**2 + 0.25*x2*(np.sin(4*x1**2)+2)**2])
        g1_x  = np.array([0,np.cos(2*x1)+2])
        g2_x  = np.array([0,np.sin(4*x1**2)+2])
        x_dot = f_x + g1_x * u1 + g2_x * u2

        return x_dot

    def stateEquation3P(state, t, input_u):
        '''
            equations 68, 69 and 70
        '''
        # current state
        x1   = state[0]
        x2   = state[1]
        # input
        u1   = input_u[0]
        u2   = input_u[1]
        u3   = input_u[2]
        # state equations
        f_x   = np.array([x2 - 2*x1, \
                -0.5*x1-x2+0.25*x2*(np.cos(2*x1)+2)**2 + 0.25*x2*(np.sin(4*x1**2)+2)**2 + 0.25*x2*(np.cos(4*x1**2)+2)**2])
        g1_x  = np.array([0,np.cos(2*x1)+2])
        g2_x  = np.array([0,np.sin(4*x1**2)+2])
        g3_x  = np.array([0,np.cos(4*x1**2)+2])
        x_dot = f_x + g1_x * u1 + g2_x * u2 + g3_x * u3

        return x_dot

    def stateEquation(self, state, t, input_u):
        if self.num_players == 2:
            return Plant2Player.stateEquation2P(state, t, input_u)
        else:
            return Plant2Player.stateEquation3P(state, t, input_u)

def main():
    # internal functions for generating optimal inputs
    def u1star(x):
        x1, x2 = x
        return -1.0 * (np.cos(2.0 * x1) + 2.0) * x2
    #
    def u2star(x):
        x1, x2 = x
        return (-1.0/2.0) * (np.sin(4.0 * x1**2) + 2.0) * x2
    # create a plant object
    # x0 = np.array([3, -1])
    x0 = np.array([1, 2])
    dt = 0.01
    sim = Plant2Player(initial_state=x0, dt=dt)
    #  iterate thru the time steps
    input_traj = []
    input_traj.append(np.array([u1star(x0), u2star(x0)]))
    for i in range(300):       #  1000 steps=10 sec; 300 steps=3 sec
        x = sim.state
        u = np.array([u1star(x), u2star(x)])
        input_traj.append(u)
        sim.nextState(u)

    print(" ")
    print("state trajectory: ", sim.state_traj)
    # print(" ")
    # print("time trajectory: ", sim.time)

    # plot the result for optimal inputs
    filename_plot = 'fig4.png'
    plot1.plot1(filename_plot, sim.time, sim.state_traj, input_traj)

    # x = np.array([1,2])
    # # u = np.array([0,0])
    # for i in range(10):
    #     x_new = x + Plant2Player.stateEquation2P(x,0,u)*sim.dt
    #     x = x_new
    #     print("x_new: ", x_new)

    # 3-player test
    # create a plant object
    x0 = np.array([1, 2])
    # x0 = np.array([1, 2])
    dt = 0.01
    sim3 = Plant2Player(initial_state=x0, dt=dt, num_players=3)
    #  iterate thru the time steps
    input_traj = []
    input_traj.append(np.array([u1star(x0), u2star(x0), 0]))    # no known solution for 3-player game
    for i in range(300):       #  1000 steps=10 sec; 300 steps=3 sec
        x = sim3.state
        u = np.array([u1star(x), u2star(x), 0])
        input_traj.append(u)
        sim3.nextState(u)

    print(" ")
    print("state trajectory: ", sim3.state_traj)

    # plot the result for optimal inputs
    filename_plot = 'fig5.png'
    plot1.plot1(filename_plot, sim3.time, sim3.state_traj, input_traj)

if __name__ == "__main__":
    main()
