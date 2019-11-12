import numpy as np

class Actor:
    # input: x (state), V_hat (value), delta_hjb (Bellman Error)
    # output: u_hat (input)
    def __init__(self, params):
        self.delta_t = params[0]
        self.R11 = params[1]
        self.R22 = params[2]
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
    def updateWeights(self):
        ''' Equations 51 '''
        W1a_hat_dot = -1.0 * self.Gamma_11a * deriv_Ea_W1a / (np.sqrt(1.0 + self.omega_1 * self.omega_1)) \
                        - self.Gamma_12a * (self.W1a_hat - self.W1c_hat) \
                        - self.Gamma_13a * self.W1a_hat
        self.W1a_hat = W1a_hat_dot * self.delta_t + self.W1a_hat
        #
        W2a_hat_dot = -1.0 * self.Gamma_21a * deriv_Ea_W2a / (np.sqrt(1.0 + self.omega_2 * self.omega_2)) \
                        - self.Gamma_22a * (self.W2a_hat - self.W2c_hat) \
                        - self.Gamma_23a * self.W2a_hat
        self.W2a_hat = W2a_hat_dot * self.delta_t + self.W2a_hat

    def policyHat(self, input):
        ''' Equations 44 '''
        # get input
        x = input[0]
        # update weights
        self.updateWeights()
        # update u1, u2
        u1_hat = (-1.0/2.0)*(1.0/self.R11) * self.g1(x) * self.phi1_prime(x) * self.W1a_hat
        u2_hat = (-1.0/2.0)*(1.0/self.R22) * self.g2(x) * self.phi2_prime(x) * self.W2a_hat
        return np.array([u1_hat], [u2_hat])
