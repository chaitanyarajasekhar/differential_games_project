import numpy as np

from src import Actor, Critic2P, Identifier, Plant2Player

def main():
    # initialize all blocks
    dt = 0.01
    plant      = Plant2Player(dt)
    identifier = Identifier(plant.state,dt)
    critic     = Critic2P(dt)
    actor      = Actor(dt)

    time_steps = 10#10000

    for i in range(time_steps):
        input_hat  = actor.policyHat(plant.state)
        next_state_hat, next_state_hat_dot = identifier.nextStateHat(plant.state,input_hat)
        next_state = plant.nextState(input_hat)

        identifier.update(next_state)
        value_function = critic.valueFunctionHat(next_state, next_state_hat_dot, input_hat)
        actor.updateWeights(next_state,critic.delta_hjb,critic.weights,critic.omega)

        # print(input_hat, value_function, next_state, next_state_hat)
        print(critic.weights)

    # print(plant.state_traj)


if __name__ == '__main__':
    main()
