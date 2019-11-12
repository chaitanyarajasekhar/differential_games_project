import numpy as np

class Plant:
    #  Plant:  input is u_hat; output is x (state)
    def __init__(self,params):
        self.delta_t = params[0]            # time increment
        self.num_players = params[1]        # number of players in game
    #
    def f1_2player(self, x):
        x1, x2 = x
        row1 = x2 - 2.0 * x1
        row2 = (-1.0/2.0) * x1 - x2 + (1.0/4.0) * x2 * (np.cos(2.0 * x1) + 2.0)**2 \
                + (1.0/4.0) * x2 * (np.sin(4.0 * x1**2) + 2.0)**2
        return np.array([[row1], [row2]])
    #
    def f1_3player(self, x):
        x1, x2 = x
        row1 = x2 - 2.0 * x1
        row2 = (-1.0/2.0) * x1 - x2 + (1.0/4.0) * x2 * (np.cos(2.0 * x1) + 2.0)**2 \
            + (1.0/4.0) * x2 * (np.sin(4.0 * x1**2) + 2.0)**2 \
            + (1.0/4.0) * x2 * (np.cos(4.0 * x1**2) + 2.0)**2
        return np.array([[row1], [row2]])
    #
    def g1(self, x):
        x1, x2 = x
        row1 = 0.0
        row2 = np.cos(2.0 * x1) + 2.0
        return np.array([[row1], [row2]])
    #
    def g2(self, x):
        x1, x2 = x
        row1 = 0.0
        row2 = np.sin(4.0 * x1**2) + 2.0
        return np.array([[row1], [row2]])
    #
    def g3(self, x):
        x1, x2 = x
        row1 = 0.0
        row2 = np.cos(4.0 * x1**2) + 2.0
        return np.array([[row1], [row2]])

    def nextState(self, input):
        ''' '''
        self.x = input[0]
        u1 = input[1]
        u2 = input[2]
        if self.num_players == 2:   # 2-player game
            x_dot = self.f1_2player(x) + self.g1(x) * u1 + self.g2(x) * u2
        else:                       # 3-player game
            u3 = input[3]
            x_dot = self.f1_3player(x) + self.g1(x) * u1 + self.g2(x) * u2 + self.g3(x) * u3
        # compute x from x_dot
        self.x = x_dot * self.delta_t + self.x
        # return
        output = self.x
        return output
