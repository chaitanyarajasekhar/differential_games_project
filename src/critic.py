import numpy as np

def costFunction(state, input_u,i):
    Q_1 = np.eye(2)
    Q_2 = 0.5 * np.eye(2)

    R   = np.array([[2,2],[1,1]])
    cost = 0
    if i == 1:
        cost = np.matmul(state,np.matmul(Q_1,state)) + R[0,0]*input_u[0]**2 +\
                            R[0,1]*input_u[1]**2
        return cost
    elif i == 2:
        cost = np.matmul(state,np.matmul(Q_2,state)) + R[1,1]*input_u[1]**2 +\
                            R[1,0]*input_u[0]**2
        return cost

    return cost

class Critic2P:
    def __init__(self, params):

        # NOTE: params has number of player informations
        self.dt = 0.0025#params[0]
        self.W1c_hat = np.array([3.0,3.0,3.0])#params[1]
        self.W2c_hat = np.array([3.0,3.0,3.0])#params[2]
        self.Gamma1c = np.array([3.0,3.0,3.0])#params[3]
        self.Gamma2c = np.array([3.0,3.0,3.0])#params[4]
        self.lambda_1 = params[5]
        self.lambda_2 = params[6]
        self.eta_1c = params[7]
        self.eta_2c = params[8]
        self.nu_1 = params[9]
        self.nu_2 = params[10]
    #
    def phi_i(x):
        x1, x2 = x
        return np.array([x1**2, x1*x2, x2**2])

    def phi_i_prime(x):
        x1, x2 = x
        return np.array([[2*x1, 0],[x2,x1],[0,2*x2]])
    #
    def updateWeights(self, state, state_hat_dot, input_u):
        '''
            equations 45, 47 and 48
        '''

        x         = state
        u         = input_u
        x_hat_dot = state_hat_dot

        # equations 45 Bellman errors calc
        r_1        = costFunction(x,u,1)
        r_2        = costFunction(x,u,2)
        omega_1    = np.matmul(self.phi_i_prime(x),x_hat_dot)
        omega_2    = np.matmul(self.phi_i_prime(x),x_hat_dot) # same as w_1
        delta_hjb1 = np.matmul(np.transpose(self.W1c_hat),omega_1)  + r_1
        delta_hjb2 = np.matmul(np.transpose(self.W2c_hat),omega_2)  + r_2

        # equations 48
        denom_eq_48_1 = 1.0+self.nu_1*np.matmul(np.transpose(omega_1),np.matmul(self.Gamma1c,omega_1))
        denom_eq_48_2 = 1.0+self.nu_2*np.matmul(np.transpose(omega_2),np.matmul(self.Gamma2c,omega_2))

        brackets_eq_48_1 = -1.0 * self.lambda_1 * self.Gamma1c +
                            np.matmul(self.Gamma1c,np.matmul(np.matmul(self.omega_1,np.transpose(self.omega_1)),\
                            self.Gamma1c))/denom_eq_48_1
        brackets_eq_48_2 = -1.0 * self.lambda_2 * self.Gamma2c +
                            np.matmul(self.Gamma2c,np.matmul(np.matmul(self.omega_2,np.transpose(self.omega_2)),\
                            self.Gamma2c))/denom_eq_48_2

        Gamma1c_dot = -1.0 * self.eta_1c * brackets_eq_48_1
        Gamma2c_dot = -1.0 * self.eta_2c * brackets_eq_48_2

        # update gamma
        self.Gamma1c = Gamma1c_dot * self.dt + self.Gamma1c
        self.Gamma2c = Gamma2c_dot * self.dt + self.Gamma2c

        # equations 47
        denom_eq_47_1 = 1.0+self.nu_1*np.matmul(np.transpose(omega_1),np.matmul(self.Gamma1c,omega_1))
        denom_eq_47_2 = 1.0+self.nu_2*np.matmul(np.transpose(omega_2),np.matmul(self.Gamma2c,omega_2))

        W1c_hat_dot = -1.0 * self.eta_1c * np.matmul(self.Gamma1c,omega_1) * delta_hjb1 \
                            / denom_eq_47_1
        W2c_hat_dot = -1.0 * self.eta_2c * np.matmul(self.Gamma2c,omega_2) * delta_hjb2 \
                            /denom_eq_47_2

        # update weights
        self.W1c_hat = W1c_hat_dot * self.dt + self.W1c_hat
        self.W2c_hat = W2c_hat_dot * self.dt + self.W2c_hat

    def valueFunctionHat(self, state, state_hat_dot, input_u):
        '''
            equations 44
        '''
        # input
        x = state
        # update weights
        self.updateWeights(state, state_hat_dot, input_u)

        # calculate value functions euqation 44
        V1_hat = np.matmul(np.transpose(self.W1c_hat),self.phi_i(x))
        V2_hat = np.matmul(np.transpose(self.W2c_hat),self.phi_i(x))

        return np.array([[V1_hat], [V2_hat]])
