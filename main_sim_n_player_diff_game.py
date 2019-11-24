import numpy as np
import matplotlib.pyplot as plt

from src import Actor, Critic2P, Identifier, Plant2Player

from datetime import datetime

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
    fig.suptitle('Time vs. State and Input', fontsize=12)
    # Top Left figure
    ax = fig.add_subplot(121)
    # ax.set_title('State', fontsize=14)
    ax.plot(time, x[:, 0], 'r')
    ax.plot(time, x[:, 1], 'b')
    ax.legend(['Player 1', 'Player2'], loc='best')
    ax.set_xlabel('Time (sec)', fontsize=14)
    ax.set_ylabel('State', fontsize=14)
    # ax.set_ylim(-0.1, 1.1)
    ax.grid()

    # Top Left figure
    ax = fig.add_subplot(122)
    # ax.set_title('Input', fontsize=14)
    ax.plot(time[0:num_timesteps-1], u[:, 0], 'r')
    ax.plot(time[0:num_timesteps-1], u[:, 1], 'b')
    ax.legend(['Player 1', 'Player2'], loc='best')
    ax.set_xlabel('Time (sec)', fontsize=14)
    ax.set_ylabel('Input', fontsize=14)
    # ax.set_ylim(-0.1, 1.1)
    ax.grid()

    plt.savefig(filename_plot)


def main():
    # initialize all blocks
    dt = 0.005
    plant      = Plant2Player(dt)
    identifier = Identifier(plant.state,dt)
    critic     = Critic2P(dt)
    actor      = Actor(dt)

    # time_steps = 1000#10000
    time_steps = 1000

    input_traj = []
    x_hat_dot_traj = []

    for i in range(time_steps):
        input_hat  = actor.policyHat(plant.state)
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

    state     = np.asarray(plant.state_traj)
    time      = np.asarray(plant.time)
    state_hat = np.asarray(identifier.state_hat_traj)
    input_array = np.asarray(input_traj)
    x_hat_dot_array = np.asarray(x_hat_dot_traj)
    # save a file full of trajectories
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    file_saved_traj = 'trajectory-' + timestamp + '.npz'
    np.savez(file_saved_traj, state=state, time=time, state_hat=state_hat, \
                input_array=input_array, x_hat_dot_array=x_hat_dot_array)

    # print(state_hat[:100])

    # plot some stuff
    filename_plot = 'plot' + timestamp + '.pdf'
    plot1(time, state, input_array, filename_plot)




if __name__ == '__main__':
    main()
