# plot1.py
#  function to help with plotting results
import pylab as plt
import numpy as np
#
def plot1(filename_plot, titles_plot, t, state, u):
    # check inputs
    print("shape of t: ", np.shape(t))
    print("type of t: ", type(t))
    print("shape of state: ", np.shape(state))
    print("type of state: ", type(state))
    print("shape of u: ", np.shape(u))
    print("type of u: ", type(u))
    array_state = np.array(state)
    array_u = np.array(u)
    # exit()
    #
    fig = plt.figure(figsize=(15, 10))
    # left figure
    ax = fig.add_subplot(121)
    ax.plot(t, array_state[:, 0])
    ax.plot(t, array_state[:, 1])
    ax.set_ylim([-10, 10])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('State')
    ax.legend(('x1', 'x2'))
    # if np.shape(u)[1] == 2:
    #     ax.set_title('2-Player Game: Time vs. State, using Optimal Input')
    # else:
    #     ax.set_title('3-Player Game: Time vs. State, with zero input for 3d Player')
    ax.set_title(titles_plot[0])
    ax.grid()
    #
    # right figure
    ax = fig.add_subplot(122)
    ax.plot(t, array_u[:, 0])
    ax.plot(t, array_u[:, 1])
    if np.shape(u)[1] > 2:
        ax.plot(t, array_u[:, 2])
    ax.set_ylim([-5, 5])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Input')
    if np.shape(u)[1] == 2:
        ax.legend(('u1 (Player1)', 'u2 (Player2)'))
        # ax.set_title('2-Player Game: Time vs. Optimal Input')
    else:
        ax.legend(('u1 (Player1)', 'u2 (Player2)', 'u3 (Player3)'))
        # ax.set_title('3-Player Game: Time vs. Input, with zero input for 3d Player')
    #
    ax.set_title(titles_plot[1])
    ax.grid()
    #
    plt.savefig(filename_plot)
