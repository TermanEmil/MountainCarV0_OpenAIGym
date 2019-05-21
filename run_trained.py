import gym
import numpy as np
from tensorflow.keras.models import load_model


# Constants
c_env_name = "MountainCar-v0"
c_max_nb_of_steps = 200

c_trained_model_file_path = './trained_models/trained_v0.h5'

# Globals
g_state_size = None
g_action_size = None

g_env = None
g_model = None


def get_action(state):
    q_values = g_model.predict(state)[0]
    return np.argmax(q_values)


def reshape_state(state):
    return np.reshape(state, [1, g_state_size])


def normalize(values, low, high):
    return (values - low) / (high - low)


def normalize_state(state):
    return normalize(state, g_env.low, g_env.high)


def preprocess_state(state):
    state = reshape_state(state)
    state = normalize_state(state)
    return state


if __name__ == '__main__':
    g_env = gym.make(c_env_name)
    g_env._max_episode_steps = c_max_nb_of_steps

    g_state_size = g_env.observation_space.shape[0]
    g_action_size = g_env.action_space.n

    g_model = load_model(c_trained_model_file_path)

    while True:
        state = preprocess_state(g_env.reset())

        for step in range(c_max_nb_of_steps):
            g_env.render()

            action = get_action(state)
            next_state, reward, done, info = g_env.step(action)
            next_state = preprocess_state(next_state)

            state = next_state

            if done:
                break
