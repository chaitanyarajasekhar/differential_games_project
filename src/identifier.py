import numpy as np
import math
from scipy.integrate import odeint


class Identifier:
    #  Identifier:  input is x_tilde; output is x_hat_dot
    def __init__(self, initial_state, dt, params=None):
        # init params
        #self.delta_t = params[0]
        self.k_gain    = 300#params[2]
        self.alpha     = 200#params[3]
        self.gamma_f   = 5#params[4]
        self.beta_1    = 0.2#params[5]
        self.dt        = dt # can change later

        # Neural Network weights initialization
        self.state_size       = 2
        self.n_hidden_neurons = 5+1
        self.W_f_hat          = 2*np.random.rand(self.n_hidden_neurons,self.state_size)-1
        self.V_f_hat          = 2*np.random.rand(self.state_size,self.n_hidden_neurons)-1
        self.Gamma_wf         = 0.1*np.eye(self.n_hidden_neurons)#params[6]
        self.Gamma_vf         = 0.1*np.eye(self.state_size)#params[7]

        self.nu = 0.0

        # NOTE:  initialize
        self.state_hat      = np.zeros(2)
        self.state_tilde_0  = initial_state - self.state_hat#np.array([-3,-1])
        self.state_hat_traj = []
        self.state_hat_traj.append(self.state_hat.copy())

        # init values of computed values
        # NOTE: not needed
        # self.sigf_hat = 0.0             # need help here
        # self.sigf_hat_old = 0.0         # need help here
        # self.x_hat_dot = 0.0

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def tanh(x):
        ''' d(tanh(x)) = (1-tanh(x)**2) '''
        # return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
        return np.tanh(x)

    def update(self,x):
        '''
            updates nu and weights
        '''
        x_tilde = x - self.state_hat
        self.updateNU(x_tilde)
        self.updateWeights(x_tilde, self.state_hat)
        pass

    #
    def updateNU(self, x_tilde):
        nu_dot = (self.k_gain * self.alpha + self.gamma_f) * x_tilde + self.beta_1 * np.sign(x_tilde)
        self.nu = nu_dot * self.dt+ self.nu
    #
    def updateWeights(self, x_tilde, x_hat):
        # NOTE: addressed # need help here
        sigma_f_hat = Identifier.tanh(np.matmul(np.transpose(self.V_f_hat),x_hat))
        sigma_f_hat_prime = np.eye(self.n_hidden_neurons)*(1-sigma_f_hat)*(1+sigma_f_hat) #(self.sigf_hat - self.sigf_hat_old) / self.delta_
        # NOTE: we have to decide if to use odeint or the approximation
        # TODO: check matrix dimensions and convert the v_f_hat_dot and w_f_hat_dot calculations using np.matmul
        # to avoid element wise multiplication

        # NOTE: had to extend the matrix dims to perform the matrix multiplication
        x_tilde   = np.expand_dims(x_tilde,axis=1)
        x_hat_dot = np.expand_dims(self.x_hat_dot,axis=1)

        # update Vf_hat
        V_f_hat_dot = self.proj(np.matmul(self.Gamma_vf,np.matmul(np.matmul(x_hat_dot,np.transpose(x_tilde)), \
                                np.matmul(np.transpose(self.W_f_hat),sigma_f_hat_prime))))
        self.V_f_hat = V_f_hat_dot * self.dt + self.V_f_hat

        # update Wf_hat
        W_f_hat_dot = self.proj(np.matmul(np.matmul(np.matmul(self.Gamma_wf, sigma_f_hat_prime),np.transpose(self.V_f_hat)),\
                                np.matmul(x_hat_dot,np.transpose(x_tilde))))
        self.W_f_hat = W_f_hat_dot * self.dt + self.W_f_hat

        # NOTE: not needed
        # update sigf_hat
        # self.sigf_hat_old = self.sigf_hat.copy()
        # self.sigf_hat = self.sigmoid(np.transpose(self.Vf_hat) * x_hat)

    def proj(self,x):
        # TODO: need to figure out this later
        return x

    def nextStateHat(self, prev_state, input_u):
        # NOTE: add args in the args
        self.x_hat_dot = Identifier.stateEquation2PHat(self.state_hat,0,
                        prev_state,input_u, self.state_tilde_0,self.W_f_hat,
                        self.V_f_hat,self.k_gain,self.nu)
        # next_state_hat = odeint(Identifier.stateEquation2PHat,self.state_hat,
        #                         np.array([0.0, self.dt]),args=(prev_state,input_u,
        #                         self.state_tilde_0,self.W_f_hat,
        #                         self.V_f_hat,self.k_gain,self.nu))

        self.state_hat = self.x_hat_dot * self.dt + self.state_hat
        # self.state_hat = next_state_hat[-1,:]

        # self.state_hat_traj.append(next_state_hat[-1,:])
        self.state_hat_traj.append(self.state_hat)

        return self.state_hat, self.x_hat_dot

    # NOTE: not needed
    # def g1(self, x):
    #     x1, x2 = x
    #     row1 = 0.0
    #     row2 = np.cos(2.0 * x1) + 2.0
    #     return np.array([[row1], [row2]])
    # #
    # def g2(self, x):
    #     x1, x2 = x
    #     row1 = 0.0
    #     row2 = np.sin(4.0 * x1**2) + 2.0
    #     return np.array([[row1], [row2]])
    # #
    # def g3(self, x):
    #     x1, x2 = x
    #     row1 = 0.0
    #     row2 = np.cos(4.0 * x1**2) + 2.0
    #     return np.array([[row1], [row2]])

    def stateEquation2PHat(state_hat,t,state,input_u, state_tilde_0,W_f_hat,V_f_hat,k_gain,nu):
        '''
            equation 18
        '''
        # NOTE: remove later
        # x       = input[0]
        # x_hat   = input[1]
        # x_tilde = x - x_hat
        # u1 = input[2]
        # u2 = input[3]
        # # update weights and nu
        # self.updateWeights(x_tilde, x_hat)
        # self.update_nu(x_tilde)
        # # calculate mu
        # mu = self.k_gain * (x_tilde - self.x_tilde_0) + self.nu

        x1        = state[0]
        x2        = state[1]
        u1        = input_u[0]
        u2        = input_u[1]
        x_hat     = state_hat
        x_tilde_0 = state_tilde_0
        x_tilde   = state - state_hat

        g1_x        = np.array([0,np.cos(2*x1)+2])
        g2_x        = np.array([0,np.sin(4*x1**2)+2])
        mu          = k_gain * (x_tilde - x_tilde_0) + nu
        sigma_f_hat = Identifier.tanh(np.matmul(np.transpose(V_f_hat),x_hat))

        x_hat_dot   = np.matmul(np.transpose(W_f_hat),sigma_f_hat) + g1_x * u1 + g2_x * u2 + mu

        # NOTE: will deal 3 player game later
        # if self.num_players == 2:
        #     self.x_hat_dot = self.Wf_hat * self.sigf_hat + self.g1(x) * u1 + self.g2(x) * u2 + mu
        # else:
        #     u3 = input[4]
        #     self.x_hat_dot = self.Wf_hat * self.sigf_hat + self.g1(x) * u1 + self.g2(x) * u2 + self.g3(x) * u3 + mu

        return x_hat_dot

def main():
    np.random.seed(40)
    id = Identifier()


    x_hat = np.zeros(2)
    x = np.array([3,1])
    input = np.array([4,4])
    id.nextStateHat(x,input)
    print(id.W_f_hat)
    print(id.V_f_hat)
    print(id.state_hat_traj)

    id.update(x)
    print(id.nu)
    print(id.W_f_hat)
    print(id.V_f_hat)

if __name__ == '__main__':
    main()
