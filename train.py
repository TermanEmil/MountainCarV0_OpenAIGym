import random
import gym
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque

# Constants
c_env_name = 'MountainCar-v0'

c_discount_rate = 0.99
c_learning_rate = 0.001

c_memory_size = 20000
c_batch_size = 64

# Required memory to start training
c_mem_len_train_start = 1000

c_exploration_max = 1.0
c_exploration_min = 0.01
c_exploration_decay = 0.9995

# Globals
g_state_size = None
g_action_size = None

g_env = None
g_memory = deque(maxlen=c_memory_size)

g_model = None
g_target_model = None
g_curriculum = None

g_epsilon = c_exploration_max

g_goal_pos = None


# General utils
def normalize(values, low, high):
    return (values - low) / (high - low)


# Reward methods
def compute_reward_state0(car_position):
    reward = 0
    if car_position >= g_goal_pos:
        reward = 10 ** 2
    elif car_position >= 0.9:
        reward = 10 ** (-1)
    elif car_position >= 0.8:
        reward = 10 ** (-2)
    elif car_position >= 0.7:
        reward = 10 ** (-3)
    elif car_position >= 0.6:
        reward = 10 ** (-4)
    elif car_position >= 0.5:
        reward = 10 ** (-6)
    elif car_position >= 0.4:
        reward = 10 ** (-7)

    return reward


def compute_reward_state1(car_position):
    reward = -1
    if car_position >= g_goal_pos:
        reward = 10 ** 2

    return reward


class TrainingCurriculum:
    max_nb_of_steps = [1500, 500, 200]
    nb_of_max_steps = len(max_nb_of_steps)

    def __init__(self):
        self._reward_f = compute_reward_state0
        self._max_nb_of_steps_i = 0
        self._learning_state = 0

    def compute_reward(self, car_position):
        return self._reward_f(car_position)

    def get_max_nb_of_steps(self):
        return TrainingCurriculum.max_nb_of_steps[self._max_nb_of_steps_i]

    def update_state(self, episodes_max_pos):
        global g_epsilon

        if len(episodes_max_pos) <= 10:
            return

        mean_pos_over_last_ten = float(np.mean(episodes_max_pos[-10:]))
        if mean_pos_over_last_ten >= g_goal_pos:
            episodes_max_pos.clear()

            initial_learning_state = self._learning_state
            if self._learning_state == 0:
                self._learning_state = 1
                self._max_nb_of_steps_i = 1

            elif self._learning_state == 1:
                self._learning_state = 2
                self._max_nb_of_steps_i = 2

            elif self._learning_state == 2:
                self._learning_state = 3
                self._reward_f = compute_reward_state1

            elif self._learning_state == 3:
                # Done training
                self._learning_state = -1

            # If learning state has changed
            if initial_learning_state != self._learning_state:
                g_epsilon = max(g_epsilon, 0.3)
                print('Learning state has changed to ', self._learning_state)

    def is_done_training(self):
        return self._learning_state == -1


# DQN
def build_model():
    model = Sequential()

    model.add(Dense(24, activation='relu', input_dim=g_state_size, kernel_initializer='he_uniform'))
    model.add(Dense(24, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(g_action_size, kernel_initializer='he_uniform'))

    model.compile(Adam(lr=c_learning_rate), 'mse')
    return model


def update_target_model():
    g_target_model.set_weights(g_model.get_weights())


def get_action(state):
    if np.random.rand() < g_epsilon:
        return g_env.action_space.sample()
    else:
        q_values = g_model.predict(state)[0]
        return np.argmax(q_values)


def append_memory(state, action, reward, next_state, done):
    global g_epsilon

    g_memory.append((state, action, reward, next_state, done))

    if g_epsilon > c_exploration_min:
        g_epsilon = max(g_epsilon * c_exploration_decay, c_exploration_min)


def can_train():
    mem_len = len(g_memory)
    return mem_len >= c_mem_len_train_start and mem_len >= c_batch_size


def train_model():
    if not can_train():
        return

    mini_batch = random.sample(g_memory, c_batch_size)

    update_input = np.zeros((c_batch_size, g_state_size))
    update_target = np.zeros((c_batch_size, g_state_size))
    action, reward, done = [], [], []

    for i in range(c_batch_size):
        update_input[i] = mini_batch[i][0]
        action.append(mini_batch[i][1])
        reward.append(mini_batch[i][2])
        update_target[i] = mini_batch[i][3]
        done.append(mini_batch[i][4])

    target = g_model.predict(update_input)
    target_next = g_model.predict(update_target)
    target_val = g_target_model.predict(update_target)

    for i in range(c_batch_size):
        if done[i]:
            target[i][action[i]] = reward[i]
        else:
            # The key point of Double DQN:
            #  Selection of action is from model
            #  Update is from target model
            a = np.argmax(target_next[i])
            target[i][action[i]] = reward[i] + c_discount_rate * target_val[i][a]

    g_model.fit(
        update_input,
        target,
        batch_size=c_batch_size,
        epochs=1,
        verbose=0)


def reshape_state(state):
    return np.reshape(state, [1, g_state_size])


def normalize_state(state):
    return normalize(state, g_env.low, g_env.high)


def preprocess_state(state):
    state = reshape_state(state)
    state = normalize_state(state)
    return state


# noinspection PyShadowingNames
def run_episode():
    done = False
    score = 0.0
    state = preprocess_state(g_env.reset())
    episode_max_pos = -999
    step = 0

    while not done:
        g_env.render()

        action = get_action(state)
        next_state, reward, done, info = g_env.step(action)
        next_state = preprocess_state(next_state)

        # Normalized car position
        car_position = state[0][0]

        done = (step >= g_curriculum.get_max_nb_of_steps()) or (car_position >= g_goal_pos)

        if car_position > episode_max_pos:
            episode_max_pos = car_position

        reward = g_curriculum.compute_reward(car_position)

        append_memory(state, action, reward, next_state, done)
        train_model()
        score += reward
        state = next_state
        step += 1

    return score, episode_max_pos, step


# Mountain car specific
def get_normalized_goal_position():
    state = g_env.observation_space.sample()
    state[0] = g_env.goal_position
    state = normalize_state(state)
    return state[0]


if __name__ == '__main__':
    g_env = gym.make(c_env_name)
    g_state_size = g_env.observation_space.shape[0]
    g_action_size = g_env.action_space.n

    g_model = build_model()
    g_target_model = build_model()
    g_curriculum = TrainingCurriculum()

    max_positions = []

    g_goal_pos = get_normalized_goal_position()

    for episode in range(10000):
        update_target_model()

        score, episode_max_pos, steps = run_episode()
        max_positions.append(episode_max_pos)

        pos_mean = float(np.mean(max_positions[-min(10, len(max_positions)):]))
        print(
            "episode: %3d, score: %03.4f, max_pos: %.2f, epsilon: %.4f, steps: %4d, pos_mean: %4.2f" %
            (episode, score, episode_max_pos, g_epsilon, steps, pos_mean))
        g_curriculum.update_state(max_positions)

        if g_curriculum.is_done_training():
            g_model.save('trained_models/trained_v0.h5')
            break
