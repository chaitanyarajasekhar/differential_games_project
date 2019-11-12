import numpy as np

class Actor:
    # input: x (state), V_hat (value), delta_hjb (Bellman Error)
    # output: u_hat (input)
    def __init__(self, params):
        # init params
        self.delta_t = params[0]
        self.R11 = params[1]
        self.R22 = params[2]
        self.R12 = params[3]
        self.R21 = params[4]
        self.Q1 = params[5]
        self.Q2 = params[6]
        self.Gamma_11a = params[7]
        self.Gamma_12a = params[8]
        self.Gamma_13a = params[9]
        self.Gamma_21a = params[10]
        self.Gamma_22a = params[11]
        self.Gamma_23a = params[12]
        x0 = params[13]
        self.update_phi1_phi2(x0)
        # init output
        self.u1_hat = 0.0
        self.u2_hat = 0.0

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
    def update_phi1_phi2(self, x):
        x1, x2 = x
        self.phi1 = [x1**2, x1*x2, x2**2]
        self.phi2 = [x1**2, x1*x2, x2**2]
    #
    def update_phi1_phi2_prime(self, x):
        phi1_old = self.phi1.copy()
        phi2_old = self.phi2.copy()
        self.update_phi1_phi2(x)
        self.phi1_prime = (self.phi1 - phi1_old) / self.delta_t
        self.phi2_prime = (self.phi2 - phi2_old) / self.delta_t
    #
    def update_omega(self, x):
        self.update_phi1_phi2_prime(x)
        #  need help here
        self.F_hat_u_hat = 0.0
        self.F_hat_u_hat = 0.0
        #
        self.omega_1 = self.phi1_prime * self.F_hat_u_hat
        self.omega_2 = self.phi2_prime * self.F_hat_u_hat
    #
    def update_local_cost(self, x):
        u1 = self.u1_hat
        u2 = self.u2_hat
        self.r1 = np.transpose(x) * self.Q1 * x + np.transpose(u1) * self.R11 * u1 + \
                    np.transpose(u2) * self.R12 * u2
        self.r2 = np.transpose(x) * self.Q2 * x + np.transpose(u2) * self.R22 * u2 + \
                    np.transpose(u1) * self.R21 * u1
    #
    def updateWeights(self, x):
        ''' Equations 50 and 51 '''
        # update G1, G2, G12, G21
        G1 = self.g1(x) * (1.0/self.R11) * self.g1(x)
        G2 = self.g2(x) * (1.0/self.R22) * self.g2(x)
        G12 = self.g2(x) * (1.0/self.R22) * self.R12 * (1.0/self.R22) * self.g2(x)
        G21 = self.g1(x) * (1.0/self.R11) * self.R21 * (1.0/self.R11) * self.g1(x)
        # update partial E_a / partial W1a_hat
        delta_hjb1 = np.transpose(self.W1c_hat) * self.omega_1 + self.r1
        delta_hjb2 = np.transpose(self.W2c_hat) * self.omega_2 + self.r1
        deriv_Ea_W1a_hat = np.transpose(self.W1a_hat - self.W1c_hat) * self.phi1_prime * \
                            G1 * np.transpose(self.phi1_prime) * delta_hjb1 + \
                            (np.transpose(self.W1a_hat) * self.phi1_prime * G21 - \
                            np.transpose(self.W2c_hat) * self.phi2_prime * G1) * \
                            np.transpose(self.phi1_prime) * delta_hjb2

        deriv_Ea_W2a_hat = (np.transpose(self.W2a_hat) * self.phi2_prime * G12 - \
                            np.transpose(self.W1c_hat) * self.phi1_prime * G2) * np.transpose(self.phi2_prime) * delta_hjb1 + \
                            np.transpose(self.W2a_hat - self.W2c_hat) * self.phi2_prime * G2 * \
                            np.transpose(self.phi2_prime) * delta_hjb2
        #
        # update W1a_hat and W2a_hat
        #  see FN3 regarding substituting for projection algorithm
        W1a_hat_dot = -1.0 * self.Gamma_11a * deriv_Ea_W1a_hat / (np.sqrt(1.0 + self.omega_1 * self.omega_1)) \
                        - self.Gamma_12a * (self.W1a_hat - self.W1c_hat) \
                        - self.Gamma_13a * self.W1a_hat
        self.W1a_hat = W1a_hat_dot * self.delta_t + self.W1a_hat
        #
        W2a_hat_dot = -1.0 * self.Gamma_21a * deriv_Ea_W2a_hat / (np.sqrt(1.0 + self.omega_2 * self.omega_2)) \
                        - self.Gamma_22a * (self.W2a_hat - self.W2c_hat) \
                        - self.Gamma_23a * self.W2a_hat
        self.W2a_hat = W2a_hat_dot * self.delta_t + self.W2a_hat

    def policyHat(self, input):
        ''' Equations 44 '''
        # get input
        x = input[0]
        # update omega - omega is the critic NN regressor vector
        self.update_omega(x)
        # update local costs r1 and r2
        self.update_local_cost(x)
        # update weights
        self.updateWeights(x)
        # update u1, u2
        self.u1_hat = (-1.0/2.0)*(1.0/self.R11) * np.transpose(self.g1(x)) * np.transpose(self.phi1_prime(x)) * self.W1a_hat
        self.u2_hat = (-1.0/2.0)*(1.0/self.R22) * np.transpose(self.g2(x)) * np.transpose(self.phi2_prime(x)) * self.W2a_hat
        output = np.array([self.u1_hat], [self.u2_hat])
        return output
