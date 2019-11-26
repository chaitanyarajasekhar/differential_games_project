import numpy as np
import matplotlib.pyplot as plt


def plot1(time, x, x_hat, u, filename_plot):
    # get shape of inputs
    print(" ")
    print("In plot1()")
    print("time.shape: ", np.shape(time))
    print("x.shape: ", np.shape(x))
    print("u.shape: ", np.shape(u))
    print("x_hat.shape",np.shape(x_hat))
    # reshape the inputs
    num_timesteps = np.shape(time)[0]
    x = np.reshape(x, [num_timesteps, 2])
    u = np.reshape(u, [num_timesteps-1, 2])
    x_hat = np.reshape(x_hat, [num_timesteps, 2])

    x_tilde =  x - x_hat
    #
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('Time vs. State, Input, State_tilde', fontsize=12)
    # Top Left figure
    ax = fig.add_subplot(311)
    # ax.set_title('State', fontsize=14)
    ax.plot(time, x[:, 0], 'r')
    ax.plot(time, x[:, 1], 'b')
    ax.legend(['x1', 'x2'], loc='best')
    ax.set_xlabel('Time (sec)', fontsize=14)
    ax.set_ylabel('State', fontsize=14)
    # ax.set_ylim(-0.1, 1.1)
    ax.grid()

    # Top Left figure
    ax = fig.add_subplot(312)
    # ax.set_title('Input', fontsize=14)
    ax.plot(time[0:num_timesteps-1], u[:, 0], 'r')
    ax.plot(time[0:num_timesteps-1], u[:, 1], 'b')
    ax.legend(['Player 1', 'Player2'], loc='best')
    ax.set_xlabel('Time (sec)', fontsize=14)
    ax.set_ylabel('Input', fontsize=14)
    # ax.set_ylim(-0.1, 1.1)
    ax.grid()

    ax = fig.add_subplot(313)
    # ax.set_title('State', fontsize=14)
    ax.plot(time, x_tilde[:, 0], 'r')
    ax.plot(time, x_tilde[:, 1], 'b')
    ax.legend(['x1_tilde', 'x2_tilde'], loc='best')
    ax.set_xlabel('Time (sec)', fontsize=14)
    ax.set_ylabel('State observation error', fontsize=14)
    # ax.set_ylim(-0.1, 1.1)
    ax.grid()

    plt.savefig(filename_plot)


def plot2(time, w1a, w2a, w1c, w2c, filename_plot):
    # get shape of inputs
    print(" ")
    print("In plot2()")
    print("time.shape: ", np.shape(time))
    print("w1a.shape: ", np.shape(w1a))
    print("w2a.shape: ", np.shape(w2a))
    print("w1c.shape",np.shape(w1c))
    print("w2c.shape",np.shape(w2c))

    # reshape the inputs
    num_timesteps = np.shape(time)[0]
    w1a = np.reshape(w1a, [num_timesteps, 3])
    w2a = np.reshape(w2a, [num_timesteps, 3])
    w1c = np.reshape(w1c, [num_timesteps, 3])
    w2c = np.reshape(w2c, [num_timesteps, 3])

    #
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('Time vs. Actor Critic weights', fontsize=12)
    # Top Left figure
    ax = fig.add_subplot(221)
    # ax.set_title('State', fontsize=14)
    ax.plot(time, w1c[:, 0], 'r')
    ax.plot(time, w1c[:, 1], 'b')
    ax.plot(time, w1c[:, 2], 'g')
    ax.legend(['w1c_1', 'w1c_2', 'w1c_3'], loc='best')
    ax.set_xlabel('Time (sec)', fontsize=14)
    ax.set_ylabel('Critic 1 weights', fontsize=14)
    # ax.set_ylim(-0.1, 1.1)
    ax.grid()
    #
    # # Top Left figure
    ax = fig.add_subplot(223)
    # ax.set_title('Input', fontsize=14)
    ax.plot(time, w2c[:, 0], 'r')
    ax.plot(time, w2c[:, 1], 'b')
    ax.plot(time, w2c[:, 2], 'g')
    ax.legend(['w2c_1', 'w2c_2', 'w2c_3'], loc='best')
    ax.set_xlabel('Time (sec)', fontsize=14)
    ax.set_ylabel('Critic 2 weights', fontsize=14)
    # ax.set_ylim(-0.1, 1.1)
    ax.grid()
    #
    ax = fig.add_subplot(222)
    # ax.set_title('State', fontsize=14)
    ax.plot(time, w1a[:, 0], 'r')
    ax.plot(time, w1a[:, 1], 'b')
    ax.plot(time, w1a[:, 2], 'g')
    ax.legend(['w1a_1', 'w1a_2', 'w1a_3'], loc='best')
    ax.set_xlabel('Time (sec)', fontsize=14)
    ax.set_ylabel('Actor 1 weights', fontsize=14)
    # ax.set_ylim(-0.1, 1.1)
    ax.grid()
    #
    ax = fig.add_subplot(224)
    # ax.set_title('State', fontsize=14)
    ax.plot(time, w2a[:, 0], 'r')
    ax.plot(time, w2a[:, 1], 'b')
    ax.plot(time, w2a[:, 2], 'g')
    ax.legend(['w2a_1', 'w2a_2', 'w2a_3'], loc='best')
    ax.set_xlabel('Time (sec)', fontsize=14)
    ax.set_ylabel('Actor 2 weights', fontsize=14)
    # ax.set_ylim(-0.1, 1.1)
    ax.grid()
    #
    plt.savefig(filename_plot)



data = np.load('trajectory-20191124153455.npz')
plot1(data['time'],data['state'],data['state_hat'],data['input_array'],"plot1-20191124153455.png")
plot2(data['time'],data['W1a_hat'],data['W2a_hat'],data['W1c_hat'],data['W2c_hat'],"plot2-20191124153455.png")
