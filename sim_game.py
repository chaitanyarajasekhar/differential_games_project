# sim_game.py
#
# python script to simulate an n-player nonzero-sum game, as seen in Johnson2015
#
# uses revision of example code for solving ODEs
# taken from https://www.gribblelab.org/compneuro/2_Modelling_Dynamical_Systems.html
#
import numpy as np
import pylab as plt
from scipy.integrate import odeint
#
# global variables
#
# plant values
R11 = 2.0
R22 = 1.0
R12 = 2.0
R21 = 1.0
Q1 = np.eye(2)
Q2 = 0.5 * np.eye(2)
# sysid gains
k_gain = 300.0
alpha = 200.0
gamma_f = 5.0
beta_1 = 0.2
Gamma_wf = 0.1 * np.eye(6)
Gamma_vf = 0.1 * np.eye(2)
# actor and critic gains
Gamma_11a = 10.0
Gamma_12a = 10.0
Gamma_13a = 10.0    # this is a guess
Gamma_21a = 20.0
Gamma_22a = 20.0
Gamma_23a = 20.0    # this is a guess
eta_1c = 50.0
eta_2c = 10.0
nu_1 = 0.001
nu_2 = 0.001
lambda_1 = 0.03
lambda_2 = 0.03
# other inits
Gamma_0 = 5000.0 * np.eye(3)
#  random number on interval [a, b] = (b - a) * random_sample() + a
state_hat_0 = [0.0, 0.0]
state_0 = [3.0, -1.0]
#
#
# FUNCTIONS
#
def omega_1():
    return
#
def Gamma1c_dot(Gamma_1c):
    return -1.0 * eta_1c * (-1.0 * lambda_1 * Gamma_1c + Gamma_1c * omega_1 * omega_1 * Gamma_1c/(1.0 + nu_1 * omega_1 * Gamma_1c * omega_1))
#
def Gamma2c_dot(Gamma_2c):
    return -1.0 * eta_2c * (-1.0 * lambda_2 * Gamma_2c + Gamma_2c * omega_2 * omega_2 * Gamma_2c/(1.0 + nu_2 * omega_2 * Gamma_2c * omega_2))
#
def W1c_hat_dot(W1c_hat):
    delta_hjb1 = W1c_hat * omega_1 + r1
    return -1.0 * eta_1c * Gamma_1c * (omega_1/(1.0 + nu_1 * omega_1 * Gamma_1c * omega_1)) * delta_hjb1
#
def W2c_hat_dot(W2c_hat):
    delta_hjb2 = W2c_hat * omega_2 + r1
    return -1.0 * eta_2c * Gamma_2c * (omega_2/(1.0 + nu_2 * omega_2 * Gamma_2c * omega_2)) * delta_hjb2
#
def W1a_hat_dot(W1a_hat):
    return -1.0 * Gamma_11a * deriv_Ea_W1a / (np.sqrt(1.0 + omega_1 * omega_1)) \
    - Gamma_12a * (W1a_hat - W1c_hat) \
    - Gamma_13a * W1a_hat
#
def W2a_hat_dot(W2a_hat):
    return -1.0 * Gamma_21a * deriv_Ea_W2a / (np.sqrt(1.0 + omega_2 * omega_2)) \
    - Gamma_22a * (W2a_hat - W2c_hat) \
    - Gamma_23a * W2a_hat
#
def phi_i(x):
    x1, x2 = x
    return [x1**2, x1*x2, x2**2]
# end of phi_i()
#
def actor(x):
    x1, x2 = x
    u1_hat = (-1.0/2.0)*(1.0/R11)*g1(x)*phi1_prime(x)*W1a_hat
    u2_hat = (-1.0/2.0)*(1.0/R22)*g2(x)*phi2_prime(x)*W2a_hat
    return [u1_hat, u2_hat]
# end of actor()
#
def critic(x):
    x1, x2 = x
    V1_hat = W1c_hat * phi_i(x)
    V2_hat = W2c_hat * phi_i(x)
    return [V1_hat, V2_hat]
# end of critic()
#
def sysid(x_hat):
    mu = k_gain * (xtilde - xtilde_0) + nu
    x_hat_dot = Wf_hat * sigf_hat + g1(x) * u1(x) + g2(x) * u2(x) + mu
    return x_hat_dot
# end of sysid()
#
def V1star(x):
    x1, x2 = x
    return (1.0/2.0)*x1**2 + x2**2
#
def V2star(x):
    x1, x2 = x
    return (1.0/4.0)*x1**2 + (1.0/2.0)*x2**2
#
def u1star(x):
    x1, x2 = x
    return -1.0 * (np.cos(2.0 * x1) + 2.0) * x2
#
def u2star(x):
    x1, x2 = x
    return (-1.0/2.0) * (np.sin(4.0 * x1**2) + 2.0) * x2
#
def u1(x):
    return actor()[0]
#
def u2(x):
    return actor()[1]
#
def f1_2player(x):
    x1, x2 = x
    row1 = x2 - 2.0 * x1
    row2 = (-1.0/2.0) * x1 - x2 + (1.0/4.0) * x2 * (np.cos(2.0 * x1) + 2.0)**2 + (1.0/4.0) * x2 * (np.sin(4.0 * x1**2) + 2.0)**2
    return np.array([[row1], [row2]])
#
def f1_3player(x):
    x1, x2 = x
    row1 = x2 - 2.0 * x1
    row2 = (-1.0/2.0) * x1 - x2 + (1.0/4.0) * x2 * (np.cos(2.0 * x1) + 2.0)**2 \
        + (1.0/4.0) * x2 * (np.sin(4.0 * x1**2) + 2.0)**2 \
        + (1.0/4.0) * x2 * (np.cos(4.0 * x1**2) + 2.0)**2
    return np.array([[row1], [row2]])
#
def g1(x):
    x1, x2 = x
    row1 = 0.0
    row2 = np.cos(2.0 * x1) + 2.0
    return np.array([[row1], [row2]])
#
def g2(x):
    x1, x2 = x
    row1 = 0.0
    row2 = np.sin(4.0 * x1**2) + 2.0
    return np.array([[row1], [row2]])
#
def g3(x):
    x1, x2 = x
    row1 = 0.0
    row2 = np.cos(4.0 * x1**2) + 2.0
    return np.array([[row1], [row2]])
#
def plant_2player(state, t):
    x1, x2 = state
    x = state
    #
    xd = f1_2player(x) + g1(x) * u1(x) + g2(x) * u2(x)
    # debug
    print("shape of xd: ", np.shape(xd))
    #
    return xd
#
def plant_3player(state, t):
    x1, x2 = state
    x = state
    #
    xd = f1_3player(x) + g1(x) * u1(x) + g2(x) * u2(x) + g3(x) * u3(x)
    # debug
    print("shape of xd: ", np.shape(xd))
    #
    return xd


#
# #  MAIN PROGRAM
#
def main():
    # init time
    t = np.arange(0, 10.0, 0.01)
    # initial state
    state0 = [3.0, -1.0]
    # integrate the state derivative to get the next state iteration
    #   usage:   next_state = odeint(<plant function>, <initial state>, <time vector>, <other args to plant function>)
    state = odeint(plant, state0, t)
    #
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(121)
    ax.plot(t,state)
    ax.set_ylim([0,8])
    ax.set_xlabel('Time')
    ax.set_ylabel('Population Size')
    ax.legend(('x (prey)','y (predator)'))
    ax.set_title('Lotka-Volterra equations')
    # plt.savefig('fig1.png')
    #
    #
    # animation in state-space
    # fig2 = plt.figure()
    ax = fig.add_subplot(122)
    pb, = ax.plot(state[:,0],state[:,1],'b-',alpha=0.2)
    ax.set_xlabel('x (prey population size)')
    ax.set_ylabel('y (predator population size)')
    p, = ax.plot(state[0:10,0],state[0:10,1],'b-')
    pp, = ax.plot(state[10,0],state[10,1],'b.',markersize=10)
    tt = ax.set_title("%4.2f sec" % 0.00)

    # animate
    step=2
    for i in range(1,np.shape(state)[0]-20,step):
        p.set_xdata(state[10+i:20+i,0])
        p.set_ydata(state[10+i:20+i,1])
        pp.set_xdata(state[19+i,0])
        pp.set_ydata(state[19+i,1])
        tt.set_text("%d steps" % (i))
        plt.draw()
     #
    plt.savefig('fig2.png')
    #

#  end of main()
##
if __name__ == "__main__":
    main()
#
