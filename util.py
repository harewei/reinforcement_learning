import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

# Plot configurations
plt.figure(1, figsize=(20, 20), dpi=80, facecolor='w', edgecolor='k')
font = {'family': 'sans-serif',
        'weight': 'normal',
        'size': 22}
plt.rc('font', **font)

rewards = []
avg_rewards = []
episode_counts = []


def visualization(episode_reward, episode_count, slide_window, filename):
    rewards.append(episode_reward)
    episode_counts.append(episode_count)
    if len(rewards) > slide_window:
        avg_rewards.append(
            avg_rewards[-1] + 1 / slide_window * (rewards[-1] - rewards[-1 - slide_window]))
    else:
        avg_rewards.append(np.sum(rewards) / len(rewards))
    print("episode: {}, reward: {}".format(episode_count, episode_reward))

    if episode_count % slide_window == 0:
        plt.subplot(2, 1, 1)
        plt.plot(episode_counts, rewards, "darkgreen")
        plt.title("Reward")
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.subplots_adjust(hspace=0.5)
        plt.subplots_adjust(wspace=0.5)
        plt.subplot(2, 1, 2)
        plt.plot(episode_counts, avg_rewards, "orange")
        plt.title("Moving Average Reward ({})".format(100))
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.savefig(filename)
