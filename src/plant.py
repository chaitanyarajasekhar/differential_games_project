import numpy as np
from scipy.integrate import odeint
# import matplotlib.pyplot as plt

class Plant2Player:
    def __init__(self,dt,seed,noise,params=None):
        '''
            intializing the equations
        '''
        self.state        = np.array([3,-1])
        self.dt           = dt # change it take the value from params
        self.time         = []
        self.state_traj   = []
        self.current_time = 0.0
        self.state_traj.append(self.state.copy())
        self.time.append(0.)
        np.random.seed(seed)
        self.noise = noise

    def g1(x):
        pass

    def g2(x):
        pass

    def nextState(self,input_u):
        '''
        '''
        # NOTE: change the input in the args
        state_dot  = Plant2Player.stateEquation2P(self.state,input_u)
        self.state = state_dot * self.dt + self.state
        # next_state = odeint(Plant2Player.stateEquation2P,self.state,np.array([0.0, self.dt]),args=(input,))

        # # NOTE: adding noise
        # self.state =  self.state + 10 * np.random.randn()/100
        self.state = self.state + self.noise

        # self.state        = next_state[-1,:]
        self.current_time = self.current_time + self.dt


        # self.state_traj.append(next_state[-1,:])
        self.state_traj.append(self.state)

        self.time.append(self.current_time)

        return self.state

    def stateEquation2P(state,input_u):
        '''
            equations 68 and 69
        '''
        # print(state)
        # current state
        x1    = state[0]
        x2    = state[1]
        u1    = input_u[0]
        u2    = input_u[1]
        # state equations
        f_x   = np.array([x2 - 2*x1, \
                -0.5*x1-x2+0.25*x2*(np.cos(2*x1)+2)**2 + 0.25*x2*(np.sin(4*x1**2)+2)**2])
        g1_x  = np.array([0,np.cos(2*x1)+2])
        g2_x  = np.array([0,np.sin(4*x1**2)+2])
        x_dot = f_x + g1_x * u1 + g2_x * u2

        return x_dot

def main():

    sim = Plant2Player(0.001)
    u = np.array([0,0])

    for i in range(2):
        sim.nextState(u)

    print(sim.state_traj)
    print(sim.time)

    # x = np.array([[1],[2]])
    # for i in range(10):
    #     x_new = x + Plant2Player.stateEquation2P(x,0,u)*0.0025
    #     x = x_new
    #     print(x_new)

if __name__ == "__main__":
    main()
