import numpy as np

class Critic:
    def __init__(self, params):

        # NOTE: params has number of player informations
        self.delta_t = params[0]
        self.W1c_hat = params[1]
        self.W2c_hat = params[2]
        self.Gamma1c = params[3]
        self.Gamma2c = params[4]
        self.lambda_1 = params[5]
        self.lambda_2 = params[6]
        self.eta_1c = params[7]
        self.eta_2c = params[8]
        self.nu_1 = params[9]
        self.nu_2 = params[10]
    #
    def phi_i(self, x):
        x1, x2 = x
        return [x1**2, x1*x2, x2**2]
    #
    def updateWeights(self):
        '''
            equations 47 and 48
        '''
        #
        Gamma1c_dot = -1.0 * self.eta_1c * \
                        (-1.0 * self.lambda_1 * self.Gamma1c + self.Gamma1c * self.omega_1 * np.transpose(self.omega_1) * \
                        self.Gamma1c/(1.0 + self.nu_1 * np.transpose(self.omega_1) * self.Gamma1c * self.omega_1))
        self.Gamma1c = Gamma1c_dot * self.delta_t + self.Gamma1c
        #
        Gamma2c_dot = -1.0 * self.eta_2c * \
                        (-1.0 * self.lambda_2 * self.Gamma2c + self.Gamma2c * self.omega_2 * np.transpose(self.omega_2) * \
                        self.Gamma2c/(1.0 + self.nu_2 * np.transpose(self.omega_2) * self.Gamma2c * self.omega_2))
        self.Gamma2c = Gamma2c_dot * self.delta_t + self.Gamma2c
        #
        delta_hjb1 = np.transpose(self.W1c_hat) * self.omega_1 + self.r1
        W1c_hat_dot = -1.0 * self.eta_1c * self.Gamma1c * \
                        (self.omega_1/(1.0 + self.nu_1 * np.transpose(self.omega_1) * self.Gamma1c * self.omega_1)) * delta_hjb1
        self.W1c_hat = W1c_hat_dot * self.delta_t + self.W1c_hat
        #
        delta_hjb2 = np.transpose(self.W2c_hat) * self.omega_2 + self.r1
        W2c_hat_dot = -1.0 * self.eta_2c * self.Gamma2c * \
                        (self.omega_2/(1.0 + self.nu_2 * np.transpose(self.omega_2) * self.Gamma2c * self.omega_2)) * delta_hjb2
        self.W2c_hat = W2c_hat_dot * self.delta_t + self.W2c_hat

    def valueFunctionHat(self, input):
        '''
            equations 44
        '''
        # get input
        x = input[0]
        # update weights
        self.updateWeights()
        # update values
        V1_hat = np.transpose(self.W1c_hat) * self.phi_i(x)
        V2_hat = np.transpose(self.W2c_hat) * self.phi_i(x)
        return np.array([[V1_hat], [V2_hat]])
