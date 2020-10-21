import numpy as np


class Logger():
    def __init__(self, slide_window):
        self.slide_window = slide_window
        self.rewards = []
        self.running_rewards = []
        self.episode_counts = []

    def log_history(self, episode_reward, episode_count):
        self.rewards.append(episode_reward)
        self.episode_counts.append(episode_count)

        if len(self.rewards) > self.slide_window:
            self.running_rewards.append(
                self.running_rewards[-1] + 1 / self.slide_window * (
                        self.rewards[-1] - self.rewards[-1 - self.slide_window]))
        else:
            self.running_rewards.append(np.sum(self.rewards) / len(self.rewards))

    def show_progress(self, t, episode_reward, episode_count):
        t.set_description("Episode: {}, Reward: {}".format(episode_count, episode_reward))
        t.set_postfix(running_reward="{:.2f}".format(self.running_rewards[-1]))