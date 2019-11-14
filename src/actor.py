import numpy as np
from critic import Critic2P, costFunction


class Actor:
    # input: x (state), V_hat (value), delta_hjb (Bellman Error)
    # output: u_hat (input)
    def __init__(self, params=None):
        # init params
        self.dt = 0.0025
        self.R11 = 2#params[1]
        self.R22 = 1#params[2]
        self.R12 = 2#params[3]
        self.R21 = 1#params[4]
        self.Q1 = np.eye(2)#params[5]
        self.Q2 = 0.5*np.eye(2)#params[6]

        self.W1a_hat  = 3.0*np.ones((3,1))#params[1]
        self.W2a_hat  = 3.0*np.ones((3,1))#params[2]

        self.Gamma_11a = 10#params[7]
        self.Gamma_12a = 10#params[8]
        self.Gamma_21a = 20#params[10]
        self.Gamma_22a = 20#params[11]

        self.policy_hist = []

        # self.Gamma_13a = params[9]
        # self.Gamma_23a = params[12]

    # NOTE: not needed
    # def update_phi1_phi2(self, x):
    #     x1, x2 = x
    #     self.phi1 = [x1**2, x1*x2, x2**2]
    #     self.phi2 = [x1**2, x1*x2, x2**2]
    # #
    # def update_phi1_phi2_prime(self, x):
    #     phi1_old = self.phi1.copy()
    #     phi2_old = self.phi2.copy()
    #     self.update_phi1_phi2(x)
    #     self.phi1_prime = (self.phi1 - phi1_old) / self.dt
    #     self.phi2_prime = (self.phi2 - phi2_old) / self.dt
    # #
    # def update_omega(self, x):
    #     self.update_phi1_phi2_prime(x)
    #     #  need help here
    #     self.F_hat_u_hat = 0.0
    #     self.F_hat_u_hat = 0.0
    #     #
    #     self.omega_1 = self.phi1_prime * self.F_hat_u_hat
    #     self.omega_2 = self.phi2_prime * self.F_hat_u_hat
    # #
    # def update_local_cost(self, x):
    #     u1 = self.u1_hat
    #     u2 = self.u2_hat
    #     self.r1 = np.transpose(x) * self.Q1 * x + np.transpose(u1) * self.R11 * u1 + \
    #                 np.transpose(u2) * self.R12 * u2
    #     self.r2 = np.transpose(x) * self.Q2 * x + np.transpose(u2) * self.R22 * u2 + \
    #                 np.transpose(u1) * self.R21 * u1


    # NOTE: later implement the g1 g2 functions in plant and call them here
    # TODO:
    def g1(self, x):
        x1, x2 = x
        return np.array([[0.0], [np.cos(2.0 * x1) + 2.0]])
    #
    def g2(self, x):
        x1, x2 = x
        return np.array([[0.0], [np.sin(4.0 * x1**2) + 2.0]])

    def phi_i_prime(self,x):
        return Critic2P.phi_i_prime(x)

    def proj(self,x):
        return x

    # #
    def updateWeights(self, state, state_hat_dot, input_u):#delta_hjb, critic_weights, omega):
        ''' Equations 50 and 51 '''

        x          = state
        # omega_1    = omega[0]
        # omega_2    = omega[1]
        # delta_hjb1 = delta_hjb[0]
        # delta_hjb2 = delta_hjb[1]
        # W1c_hat    = critic_weights[0]
        # W2c_hat    = critic_weights[1]

        # NOTE:  only for testing
        u         = input_u
        x_hat_dot = state_hat_dot
        # equations 45 Bellman errors calc
        r_1        = costFunction(x,u,1)
        r_2        = costFunction(x,u,2)
        W1c_hat  = 3.0*np.ones((3,1))#params[1]
        W2c_hat  = 3.0*np.ones((3,1))#params[2]
        omega_1    = np.expand_dims(np.matmul(self.phi_i_prime(x),x_hat_dot),axis=1)
        omega_2    = np.expand_dims(np.matmul(self.phi_i_prime(x),x_hat_dot),axis=1) # same as w_1
        delta_hjb1 = np.matmul(np.transpose(W1c_hat),omega_1)  + r_1
        delta_hjb2 = np.matmul(np.transpose(W2c_hat),omega_2)  + r_2

        # update G1, G2, G12, G21
        G1 = np.matmul(self.g1(x),(1.0/self.R11) * self.g1(x).transpose())
        G2 = np.matmul(self.g2(x),(1.0/self.R22) * self.g2(x).transpose())
        G12 = np.matmul(self.g2(x) * (1.0/self.R22) * self.R12 *(1.0/self.R22),self.g2(x).transpose())
        G21 = np.matmul(self.g1(x) * (1.0/self.R11) * self.R21 * (1.0/self.R11),self.g1(x).transpose())

        # equations 50 setup
        phi_i_prime_x = self.phi_i_prime(x)
        brackets_50_1 = np.matmul(np.matmul(self.W1a_hat.transpose(),phi_i_prime_x),G21) -\
                            np.matmul(np.matmul(W2c_hat.transpose(),phi_i_prime_x),G1)
        brackets_50_2 = np.matmul(np.matmul(self.W2a_hat.transpose(),phi_i_prime_x),G12) -\
                            np.matmul(np.matmul(W1c_hat.transpose(),phi_i_prime_x),G2)
        # equations 50
        deriv_Ea_W1a_hat = np.matmul(np.matmul(np.matmul(np.transpose(self.W1a_hat - W1c_hat),
                                phi_i_prime_x),G1),np.transpose(phi_i_prime_x)) * delta_hjb1 + \
                                np.matmul(brackets_50_1,phi_i_prime_x.transpose()) * delta_hjb2

        deriv_Ea_W2a_hat = np.matmul(brackets_50_2,phi_i_prime_x.transpose()) * delta_hjb1 + \
                            np.matmul(np.matmul(np.matmul(np.transpose(self.W2a_hat-W2c_hat),
                            phi_i_prime_x),G2),phi_i_prime_x.transpose()) * delta_hjb2

        # TODO: see FN3 regarding substituting for projection algorithm
        # equation 51
        W1a_hat_dot = self.proj(-1.0 * self.Gamma_11a * deriv_Ea_W1a_hat.transpose() /\
                        (np.sqrt(1.0 + np.matmul(omega_1.transpose(),omega_1))) \
                        - self.Gamma_12a * (self.W1a_hat - W1c_hat))
        W2a_hat_dot = self.proj(-1.0 * self.Gamma_21a * deriv_Ea_W2a_hat.transpose() /\
                        (np.sqrt(1.0 + np.matmul(omega_2.transpose(),omega_2))) \
                        - self.Gamma_22a * (self.W2a_hat - W2c_hat))

        # update W1a_hat and W2a_hat
        self.W1a_hat = W1a_hat_dot * self.dt + self.W1a_hat
        self.W2a_hat = W2a_hat_dot * self.dt + self.W2a_hat

    def policyHat(self, state,x_hat_dot,u, update_weights = True):
        ''' Equations 44 '''
        # get input
        x = state
        if update_weights == True:
            self.updateWeights(x,x_hat_dot,u)

        # calculate u1, u2
        u1_hat = -0.5*(1.0/self.R11)*np.matmul(np.matmul(np.transpose(self.g1(x)),np.transpose(self.phi_i_prime(x))),self.W1a_hat)
        u2_hat = -0.5*(1.0/self.R22)*np.matmul(np.matmul(np.transpose(self.g2(x)),np.transpose(self.phi_i_prime(x))),self.W2a_hat)
        output = np.array([np.squeeze(u1_hat),np.squeeze(u2_hat)])
        self.policy_hist.append(output)

        return output

def main():
    act = Actor()
    x_hat_dot = np.array([-0.1,-0.1])
    x         = np.array([3,1])
    u         = np.array([4,1])
    print(act.policyHat(x,x_hat_dot,u))

if __name__ == '__main__':
    main()
