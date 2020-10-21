import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt


# Plot configurations
plt.figure(1, figsize=(20, 20), dpi=80, facecolor='w', edgecolor='k')
font = {'family': 'sans-serif',
        'weight': 'normal',
        'size': 22}
plt.rc('font', **font)


def visualize(rewards, avg_rewards, episode_counts, filename):
    plt.subplot(2, 1, 1)
    plt.plot(episode_counts, rewards, "darkgreen")
    plt.title("Episode Reward")
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
