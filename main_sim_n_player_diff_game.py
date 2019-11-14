import numpy as np

from src import Actor, Critic2P, Identifier, Plant2Player

def main():
    # initialize all blocks
    plant      = Plant2Player()
    identifier = Identifier(plant.state)
    critic     = Critic2P()
    actor      = Actor()

    time_steps = 10#10000

    for i in range(time_steps):
        input_hat  = actor.policyHat(plant.state)
        next_state_hat, next_state_hat_dot = identifier.nextStateHat(plant.state,input_hat)
        next_state = plant.nextState(input_hat)

        identifier.update(next_state)
        _ = critic.valueFunctionHat(next_state, next_state_hat, input_hat)
        actor.updateWeights(next_state,critic.delta_hjb,critic.weights,critic.omega)

    print(plant.state_traj)


if __name__ == '__main__':
    main()
