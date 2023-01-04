import numpy as np
import matplotlib.pyplot as plt


episode = 300
window_size = episode / 10


def move_avg(data):
    window = np.ones(int(window_size)) / float(window_size)

    return np.convolve(data, window, 'same')


def draw_env_record():
    env_record = {'success_rate': [], 'usage': [], 'reward': []}
    for key in env_record.keys():
        env_record[key].extend(move_avg(np.load(f'record/env/{key}_{episode}.npy', allow_pickle=True)))
    x = np.arange(len(env_record['success_rate']))
    fig, plot = plt.subplots(len(env_record))
    for idx, key in zip(range(len(env_record)), env_record.keys()):
        plot[idx].plot(x, env_record[key], 'r')
        plot[idx].set_xlim(int(window_size / 2), int(episode - (window_size / 2)))
        plot[idx].grid(True)
        plot[idx].set_title(f'{key}')
    fig.tight_layout()
    plt.show()


def draw_training_record():
    y = move_avg(np.load(f'record/loss_{episode}.npy'))
    x = np.arange(len(y))
    plt.plot(x, y, 'r')
    plt.xlim(int(window_size / 2), int(episode - (window_size / 2)))
    plt.title('Loss')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    draw_env_record()
    draw_training_record()