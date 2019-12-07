import numpy as np
import matplotlib.pyplot as plt

from src import Actor, Critic2P, Identifier, Plant2Player

from datetime import datetime
import random

# set random seeds
seed = 47           #  42 for state convergence in ~3 secs; 44 (6 secs); 47 (2 secs)
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
# set noise params
# noise_multiplier = 0.00000001                                      # sigma
noise_multiplier = 0.0
noise_variance = noise_multiplier * noise_multiplier        # sigma^2
# compute noise
noise = noise_multiplier * np.random.randn()

def plot1(time, x, u, filename_plot):
    # get shape of inputs
    print(" ")
    print("In plot1()")
    print("time.shape: ", np.shape(time))
    print("x.shape: ", np.shape(x))
    print("u.shape: ", np.shape(u))
    # reshape the inputs
    num_timesteps = np.shape(time)[0]
    x = np.reshape(x, [num_timesteps, 2])
    u = np.reshape(u, [num_timesteps-1, 2])
    #
    fig = plt.figure(figsize=(15, 10))
    # fig.suptitle('Time vs. State and Input for seed={}, noise_var={}'.format(seed, noise_variance), fontsize=12)
    # Top Left figure
    ax = fig.add_subplot(121)
    # ax.set_title('State', fontsize=14)
    ax.plot(time, x[:, 0], 'r')
    ax.plot(time, x[:, 1], 'b')
    ax.legend(['x1', 'x2'], loc='best')
    ax.set_xlabel('Time (sec)', fontsize=14)
    ax.set_ylabel('State', fontsize=14)
    # ax.set_ylim(-0.1, 1.1)
    ax.grid()

    # Top Left figure
    ax = fig.add_subplot(122)
    # ax.set_title('Input', fontsize=14)
    ax.plot(time[0:num_timesteps-1], u[:, 0], 'r')
    ax.plot(time[0:num_timesteps-1], u[:, 1], 'b')
    ax.legend(['Player1', 'Player2'], loc='best')
    ax.set_xlabel('Time (sec)', fontsize=14)
    ax.set_ylabel('Input', fontsize=14)
    # ax.set_ylim(-0.1, 1.1)
    ax.grid()

    plt.savefig(filename_plot)

def exploratorySignal(t, ex_enable):
    if t < 6.0 and ex_enable == True:
        return np.sin(5*np.pi*t)+np.sin(np.e*t)+np.sin(t)**5+np.cos(20*t)**5 +\
                (np.sin(-1.2*t)**2)*np.cos(0.5*t)
    else:
        return 0.0


def main():
    # initialize all blocks
    dt = 0.005
    plant      = Plant2Player(dt, seed, noise)
    identifier = Identifier(plant.state, dt, seed)
    critic     = Critic2P(dt)
    actor      = Actor(dt)

    # time_steps = 1000#10000
    time_steps = 2*1000             # 10 secs
    # time_steps = 10*1000            # 50 secs

    input_traj = []
    x_hat_dot_traj = []
    ex_enable = True

    for i in range(time_steps):
        # exploratory signal
        n_t        = exploratorySignal(plant.current_time, ex_enable)
        input_hat  = actor.policyHat(plant.state) + n_t
        next_state_hat, next_state_hat_dot = identifier.nextStateHat(plant.state,input_hat)
        next_state = plant.nextState(input_hat)

        # debug
        print(" ")
        print("time step=", i)
        print("identifier.W_f_hat: ", identifier.W_f_hat)
        print("identifier.V_f_hat: ", identifier.V_f_hat)
        print("identifier.x_hat_dot: ", identifier.x_hat_dot)
        # print(i,identifier.W_f_hat,identifier.V_f_hat,identifier.x_hat_dot)#identifier.x_hat_dot,plant.state,identifier.state_hat)

        identifier.update(next_state)
        value_function = critic.valueFunctionHat(next_state, next_state_hat_dot, input_hat)
        actor.updateWeights(next_state,critic.delta_hjb,critic.weights,critic.omega)


        print("critic.delta_hjb: ", critic.delta_hjb)
        print("value_function: ", value_function)
        print("input_hat: ", input_hat)

        input_traj.append(input_hat)
        x_hat_dot_traj.append(next_state_hat_dot)
        # print(critic.delta_hjb,value_function,input_hat)#, value_function, next_state, next_state_hat)
        # print(actor.W2a_hat)
        # print(i,identifier.W_f_hat,identifier.V_f_hat.identifier.x_hat_dot,plant.state,identifier.state_hat)

    state           = np.asarray(plant.state_traj)
    time            = np.asarray(plant.time)
    state_hat       = np.asarray(identifier.state_hat_traj)
    input_array     = np.asarray(input_traj)
    x_hat_dot_array = np.asarray(x_hat_dot_traj)
    W1a_hat         = np.asarray(actor.W1a_hat_hist)
    W2a_hat         = np.asarray(actor.W2a_hat_hist)
    W1c_hat         = np.asarray(critic.W1c_hat_hist)
    W2c_hat         = np.asarray(critic.W2c_hat_hist)

    # save a file full of trajectories
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    file_saved_traj = 'trajectory-' + timestamp + '.npz'
    np.savez(file_saved_traj, state=state, time=time, state_hat=state_hat, \
                input_array=input_array, x_hat_dot_array=x_hat_dot_array, \
                W1a_hat = W1a_hat, W2a_hat=W2a_hat, W1c_hat=W1c_hat,\
                W2c_hat=W2c_hat, seed=seed, noise_variance=noise_variance, ex_enable=ex_enable)

    # print(state_hat[:100])

    # plot some stuff
    filename_plot = 'plot' + timestamp + '.pdf'
    plot1(time, state, input_array, filename_plot)




if __name__ == '__main__':
    main()
