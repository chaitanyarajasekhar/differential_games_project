import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# def stateEquation2P(state,t,input_1,input_2):
#     '''
#         equations 68 and 69
#     '''
#     # current state
#     x1   = state[0]
#     x2   = state[1]
#     # state equations
#     f_x   = np.array([x2 - 2*x1, \
#             -0.5*x2+0.25*x2*(np.cos(2*x1)+2)**2 + 0.25*x2*(np.sin(4*x1**2)+2)**2])
#     g1_x  = np.array([0,np.cos(2*x1)+2])
#     g2_x  = np.array([0,np.sin(4*x1**2)+2])
#     x_dot = f_x + g1_x * input_1 + g2_x * input_2
#
#     return x_dot

class Plant2Player:
    def __init__(self,params=None):
        '''
            intializing the equations
        '''
        self.state        = np.array([1,2])
        self.dt           = 0.0025 # change it take the value from params
        self.time         = []
        self.state_traj   = []
        self.current_time = 0.0
        self.state_traj.append(self.state.copy())
        self.time.append(0.)

    def nextState(self,input=None):
        '''
        '''
        # NOTE: change the input in the args
        next_state = odeint(Plant2Player.stateEquation2P,self.state,np.array([0.0, self.dt]),args=([0,0],))

        self.state        = next_state[-1,:]
        self.current_time = self.current_time + self.dt

        self.state_traj.append(next_state[-1,:])
        self.time.append(self.current_time)

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
                -0.5*x2+0.25*x2*(np.cos(2*x1)+2)**2 + 0.25*x2*(np.sin(4*x1**2)+2)**2])
        g1_x  = np.array([0,np.cos(2*x1)+2])
        g2_x  = np.array([0,np.sin(4*x1**2)+2])
        x_dot = f_x + g1_x * u1 + g2_x * u2

        return x_dot

def main():

    sim = Plant2Player()

    for i in range(10):
        sim.nextState()

    print(sim.state_traj)
    print(sim.time)

if __name__ == "__main__":
    main()
