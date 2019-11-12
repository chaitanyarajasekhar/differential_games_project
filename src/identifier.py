import numpy as np
import math

class Identifier:
    #  Identifier:  input is x_tilde; output is x_hat_dot
    def __init__(self, params):
        # init params
        self.delta_t = params[0]
        self.x_tilde_0 = params[1]
        self.k_gain = params[2]
        self.alpha = params[3]
        self.gamma_f = params[4]
        self.beta_1 = params[5]
        self.Gamma_wf = params[6]
        self.Gamma_vf = params[7]
        self.Wf_hat = params[8]         # need help here
        self.Vf_hat = params[9]         # need help here
        self.nu = 0.0
        # init values of computed values
        self.sigf_hat = 0.0             # need help here
        self.sigf_hat_old = 0.0         # need help here 
        self.x_hat_dot = 0.0
    #
    def sigmoid(self, x):
        return 1.0 / (1.0 + math.exp(-x))
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
        nu_dot = (self.k_gain * self.alpha + self.gamma_f) * x_tilde + self.beta_1 * np.sign(x_tilde)
        self.nu = nu_dot * self.delta_t + self.nu
    #
    def updateWeights(self, x_tilde, x_hat):
        # need help here
        self.sigf_hat_prime = (self.sigf_hat - self.sigf_hat_old) / self.delta_t
        # update Vf_hat
        Vf_hat_dot = self.proj( self.Gamma_vf * self.x_hat_dot * np.transpose(x_tilde) * \
                                np.transpose(self.Wf_hat) * self.sigf_hat_prime )
        self.Vf_hat = Vf_hat_dot * self.delta_t + self.Vf_hat
        # update Wf_hat
        Wf_hat_dot = self.proj( self.Gamma_wf * self.sigf_hat_prime * np.transpose(self.Vf_hat) * \
                                self.x_hat_dot * np.transpose(x_tilde) )
        self.Wf_hat = Wf_hat_dot * self.delta_t + self.Wf_hat
        #
        # update sigf_hat
        self.sigf_hat_old = self.sigf_hat.copy()
        self.sigf_hat = self.sigmoid(np.transpose(self.Vf_hat) * x_hat)

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
        self.updateWeights(x_tilde, x_hat)
        self.update_nu(x_tilde)
        # calculate mu
        mu = self.k_gain * (x_tilde - self.x_tilde_0) + self.nu
        if self.num_players == 2:
            self.x_hat_dot = self.Wf_hat * self.sigf_hat + self.g1(x) * u1 + self.g2(x) * u2 + mu
        else:
            u3 = input[4]
            self.x_hat_dot = self.Wf_hat * self.sigf_hat + self.g1(x) * u1 + self.g2(x) * u2 + self.g3(x) * u3 + mu
        #
        # return
        return self.x_hat_dot
