import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('results/training_log.csv')
plt.plot(df['episode'], df['reward'], label='Reward')
plt.plot(df['episode'], df['sr'], label='SR')
plt.xlabel('Episode')
plt.ylabel('Values')
plt.title('Logging Timeline Plot')
plt.legend()
plt.savefig('logging_timeline.png')
plt.close()