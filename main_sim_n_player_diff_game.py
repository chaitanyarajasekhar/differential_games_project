import numpy as np
import matplotlib.pyplot as plt

from src import Actor, Critic2P, Identifier, Plant2Player

def main():
    # initialize all blocks
    dt = 0.005
    plant      = Plant2Player(dt)
    identifier = Identifier(plant.state,dt)
    critic     = Critic2P(dt)
    actor      = Actor(dt)

    time_steps = 1000#10000

    for i in range(time_steps):
        input_hat  = actor.policyHat(plant.state)
        next_state_hat, next_state_hat_dot = identifier.nextStateHat(plant.state,input_hat)
        next_state = plant.nextState(input_hat)

        print(i,identifier.W_f_hat,identifier.V_f_hat,identifier.x_hat_dot)#identifier.x_hat_dot,plant.state,identifier.state_hat)

        identifier.update(next_state)
        value_function = critic.valueFunctionHat(next_state, next_state_hat_dot, input_hat)
        actor.updateWeights(next_state,critic.delta_hjb,critic.weights,critic.omega)

        print(critic.delta_hjb,value_function,input_hat)#, value_function, next_state, next_state_hat)
        # print(actor.W2a_hat)
        # print(i,identifier.W_f_hat,identifier.V_f_hat.identifier.x_hat_dot,plant.state,identifier.state_hat)

    state     = np.asarray(plant.state_traj)
    time      = np.asarray(plant.time)
    state_hat = np.asarray(identifier.state_hat_traj)

    # print(state_hat[:100])

    # fig, axs = plt.subplots(2)
    # sample=100
    # axs[0].plot(time[::sample],state[::sample,0],time[::sample],state[::sample,1])
    # axs[1].plot(time[::sample],state_hat[::sample,0],time[::sample],state_hat[::sample,1])
    # #
    # plt.savefig("figures/test.pdf")


if __name__ == '__main__':
    main()
