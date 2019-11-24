import numpy as np
import math
from scipy.integrate import odeint


class Identifier:
    #  Identifier:  input is x_tilde; output is x_hat_dot
    def __init__(self, initial_state, dt, params=None):
        # init params
        self.k_gain    = 300#params[2]
        self.k_gain    = 0.03  # experimental
        self.alpha     = 200#params[3]
        self.gamma_f   = 5#params[4]
        self.beta_1    = 0.2#params[5]
        self.dt        = dt # can change later

        # Neural Network weights initialization
        self.state_size       = 2
        self.n_hidden_neurons = 5
        self.W_f_hat          = 2*np.random.rand(self.n_hidden_neurons+1,self.state_size)-1
        self.V_f_hat          = 2*np.random.rand(self.state_size,self.n_hidden_neurons)-1
        self.Gamma_wf         = 0.1*np.eye(self.n_hidden_neurons+1)#params[6]
        self.Gamma_vf         = 0.1*np.eye(self.state_size)#params[7]

        self.nu = 0.0

        # NOTE:  initialize
        self.state_hat      = np.zeros(2)
        self.state_tilde_0  = initial_state - self.state_hat#np.array([-3,-1])
        self.state_hat_traj = []
        self.state_hat_traj.append(self.state_hat.copy())

        self.sigma_W_f = 100
        self.sigma_V_f = 100

        W_f_hat_lb = -3*np.ones((self.n_hidden_neurons+1,self.state_size))
        W_f_hat_ub = 3*np.ones((self.n_hidden_neurons+1,self.state_size))
        V_f_hat_lb = -3*np.ones((self.state_size,self.n_hidden_neurons))
        V_f_hat_ub = 3*np.ones((self.state_size,self.n_hidden_neurons))

        self.W_f_hat_b  = []
        self.V_f_hat_b  = []
        self.W_f_hat_b.append(W_f_hat_lb)
        self.W_f_hat_b.append(W_f_hat_ub)
        self.V_f_hat_b.append(V_f_hat_lb)
        self.V_f_hat_b.append(V_f_hat_ub)


    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def tanh(x):
        ''' d(thah(x)) = (1-tanh(x)**2) '''
        return np.tanh(x)#(np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

    def update(self,x):
        '''
            updates nu and weights
        '''
        x_tilde = x - self.state_hat
        self.updateNU(x_tilde)
        self.updateWeights(x_tilde, self.state_hat)

    #
    def updateNU(self, x_tilde):
        nu_dot = (self.k_gain * self.alpha + self.gamma_f) * x_tilde + self.beta_1 * np.sign(x_tilde)
        self.nu = nu_dot * self.dt+ self.nu
    #
    def updateWeights(self, x_tilde, x_hat):
        #
        # x_hat_v     = np.expand_dims(np.concatenate((np.ones(1),x_hat),axis=0),axis=1)
        x_hat_v     = np.expand_dims(x_hat,axis=1)
        sigma_f_hat = Identifier.tanh(np.matmul(np.transpose(self.V_f_hat),x_hat_v))

        sigma_f_hat_prime   = np.matmul((1-sigma_f_hat),np.transpose((1+sigma_f_hat))) #(self.sigf_hat - self.sigf_hat_old) / self.delta_
        sigma_f_hat_prime_v = np.concatenate((np.ones((1,self.n_hidden_neurons)),sigma_f_hat_prime), axis = 0)

        # NOTE: had to extend the matrix dims to perform the matrix multiplication
        x_tilde   = np.expand_dims(x_tilde,axis=1)
        x_hat_dot = np.expand_dims(self.x_hat_dot,axis=1)

        # print(sigma_f_hat_prime_v)
        # print(self.W_f_hat)
        # print(x_tilde,x_hat_dot)

        # calculate  Vf_hat_dot and W_f_hat_dot
        V_f_hat_dot = self.proj(np.matmul(self.Gamma_vf,np.matmul(np.matmul(x_hat_dot,np.transpose(x_tilde)), \
                                np.matmul(np.transpose(self.W_f_hat),sigma_f_hat_prime_v))),
                                self.V_f_hat,self.sigma_V_f,self.V_f_hat_b)
        W_f_hat_dot = self.proj(np.matmul(np.matmul(np.matmul(self.Gamma_wf, sigma_f_hat_prime_v),np.transpose(self.V_f_hat)),\
                                np.matmul(x_hat_dot,np.transpose(x_tilde))),
                                self.W_f_hat,self.sigma_W_f,self.W_f_hat_b)

        # update Wf_hat V_f_hat
        self.V_f_hat = V_f_hat_dot * self.dt + self.V_f_hat
        self.W_f_hat = W_f_hat_dot * self.dt + self.W_f_hat

        # experimental
        self.V_f_hat = np.minimum(np.maximum(self.V_f_hat, self.V_f_hat_b[0]), self.V_f_hat_b[1])
        self.W_f_hat = np.minimum(np.maximum(self.W_f_hat, self.W_f_hat_b[0]), self.W_f_hat_b[1])

    def proj(self,theta_hat_dot,theta_hat,sigma=None,theta_bounds=None):
        # TODO: need to figure out this later
        # continousProj()
        # return Identifier.sigmaModification(theta_hat_dot,theta_hat,sigma)
        return Identifier.continousProj(theta_hat,theta_hat_dot, theta_bounds)

    def sigmaModification(theta_dot, theta_hat, sigma):
        ''' sigma modification'''
        return theta_dot - sigma*theta_hat

    def continousProj(theta_hat,theta_hat_dot, theta_bounds):
        ''' simple projection algorithm '''
        # if theta_hat > theta_hat_lb and theta_hat < theta_hat_ub
        # or
        # theta_hat >= theta_hat_lb and theta_hat_dot >=   0
        # or
        # theta_hat <= theta_hat_ub and theta_hat_dot <= 0
        # then projection will be theta_dot else 0
        theta_hat_lb = theta_bounds[0]
        theta_hat_ub = theta_bounds[1]

        cond_1 = np.logical_and(theta_hat > theta_hat_lb,theta_hat < theta_hat_ub)
        cond_2 = np.logical_and(theta_hat >= theta_hat_lb, theta_hat_dot >= 0)
        cond_3 = np.logical_and(theta_hat <= theta_hat_ub,theta_hat_dot <= 0)

        cond = np.logical_or(np.logical_or(cond_1,cond_2),cond_3).astype(np.float64)
        theta_hat_dot = cond * theta_hat_dot

        return theta_hat_dot

    # def eModification():
    #     pass

    def nextStateHat(self, prev_state, input_u):
        # calc x_hat_dot
        self.x_hat_dot = self.stateEquation2PHat(prev_state,input_u)
        self.state_hat = self.x_hat_dot * self.dt + self.state_hat
        self.state_hat_traj.append(self.state_hat)

        return self.state_hat, self.x_hat_dot

    # # TODO: import g1 and g2 from plant NOTE: not needed
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

    def stateEquation2PHat(self,state,input_u):
        '''
            equation 18
        '''

        x1        = state[0]
        x2        = state[1]
        u1        = input_u[0]
        u2        = input_u[1]
        x_hat_v   = self.state_hat#np.concatenate((np.array([1]),self.state_hat),axis=0)
        x_tilde_0 = self.state_tilde_0
        x_tilde   = state - self.state_hat

        g1_x          = np.array([0,np.cos(2*x1)+2])
        g2_x          = np.array([0,np.sin(4*x1**2)+2])
        mu            = self.k_gain * (x_tilde - x_tilde_0) + self.nu
        sigma_f_hat   = Identifier.tanh(np.matmul(np.transpose(self.V_f_hat),x_hat_v))
        sigma_f_hat_v = np.concatenate((np.array([1]),sigma_f_hat),axis=0)

        x_hat_dot   = np.matmul(np.transpose(self.W_f_hat),sigma_f_hat_v) + g1_x * u1 + g2_x * u2 + mu

        # debug
        print(" ")
        print("In Identifier()")
        print("x_hat_dot: ", x_hat_dot)
        print("W_f_hat: ", self.W_f_hat)
        print("sigma_f_hat_v: ", sigma_f_hat_v)
        print("g1_x: ", g1_x)
        print("u1: ", u1)
        print("g2_x: ", g2_x)
        print("u2: ", u2)
        print("mu: ", mu)
        print("self.nu: ", self.nu)
        print("x_tilde: ", x_tilde)
        print("x_tilde_0: ", x_tilde_0)
        print("x_hat: ", x_hat_v)

        # NOTE: will deal 3 player game later
        # if self.num_players == 2:
        #     self.x_hat_dot = self.Wf_hat * self.sigf_hat + self.g1(x) * u1 + self.g2(x) * u2 + mu
        # else:
        #     u3 = input[4]
        #     self.x_hat_dot = self.Wf_hat * self.sigf_hat + self.g1(x) * u1 + self.g2(x) * u2 + self.g3(x) * u3 + mu

        return x_hat_dot

def main():
    np.random.seed(40)
    x = np.array([3,-1])
    input_u = np.array([0,0])

    id = Identifier(x,0.001)
    next_state = np.array([2.95,-0.95])
    id.nextStateHat(next_state,input_u)

    print(id.W_f_hat)
    print(id.V_f_hat)
    # print(id.state_hat_traj)

    id.update(np.array([2.92,-0.92]))
    print(id.nu)
    print(id.W_f_hat)
    print(id.V_f_hat)

if __name__ == '__main__':
    main()
