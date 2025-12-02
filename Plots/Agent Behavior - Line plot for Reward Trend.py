import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('results/training_log.csv')
plt.plot(df['episode'], df['reward'], label='Reward')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward Trend Over Episodes')
plt.legend()
plt.savefig('reward_trend_line.png')
plt.close()