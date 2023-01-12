import numpy as np
import matplotlib.pyplot as plt


episode = 1500
window_size = episode / 20


def move_average(data):
    return np.convolve(data, np.ones(int(window_size)) / float(window_size), 'same')


def draw():
    # draw the record of the environment
    env_record = {'success_rate': [], 'usage': [], 'reward': []}
    for key in env_record.keys():
        env_record[key].extend(move_average(np.load(f'record/env/{key}_{episode}.npy', allow_pickle=True)))
        # env_record[key].extend(move_average(np.load(f'history/0111_3/env/{key}_{episode}.npy', allow_pickle=True)))
    x = np.arange(len(env_record['success_rate']))
    fig, plot = plt.subplots(len(env_record))
    for idx, key in zip(range(len(env_record)), env_record.keys()):
        plot[idx].plot(x, env_record[key], 'r')
        plot[idx].set_xlim(int(window_size / 2), int(episode - (window_size / 2)))
        plot[idx].grid(True)
        plot[idx].set_title(f'{key}')
    fig.tight_layout()

    # draw the record of training
    y = move_average(np.load(f'record/loss_{episode}.npy'))
    # y = move_average(np.load(f'history/0111_3/loss_{episode}.npy'))
    x = np.arange(len(y))
    plt.figure()
    plt.plot(x, y, 'r')
    plt.xlim(int(window_size / 2), int(episode - (window_size / 2)))
    plt.title('Loss')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    draw()