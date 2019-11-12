import numpy as np

class Identifier:
    #  Identifier:  input is x_tilde; output is x_hat_dot
    def __init__(self, params):
        self.x_tilde_0 = params[0]
        self.k_gain = params[1]
        self.alpha = params[2]
        self.gamma = params[3]
        self.beta1 = params[4]
        self.delta_t = params[5]
        self.nu = 0.0
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
    #
    def update_nu(self, x_tilde):
        nu_dot = (self.k_gain * self.alpha + self.gamma) * x_tilde + self.beta1 * np.sign(x_tilde)
        self.nu = nu_dot * self.delta_t + self.nu
    #
    def updateWeights(self):
        self.Wf_hat = self.Wf_hat
        self.sigf_hat = self.sigf_hat

    def nextStateHat(self, input):
        '''
            equation 18
        '''
        x = input[0]
        x_hat = input[1]
        x_tilde = x - x_hat
        u1 = input[2]
        u2 = input[3]
        # update weights and nu
        self.updateWeights()
        self.update_nu()
        # calculate mu
        mu = self.k_gain * (x_tilde - self.x_tilde_0) + self.nu
        if self.num_players == 2:
            x_hat_dot = self.Wf_hat * self.sigf_hat + self.g1(x) * u1 + self.g2(x) * u2 + mu
        else:
            u3 = input[4]
            x_hat_dot = self.Wf_hat * self.sigf_hat + self.g1(x) * u1 + self.g2(x) * u2 + self.g3(x) * u3 + mu
        #
        # return
        return x_hat_dot
