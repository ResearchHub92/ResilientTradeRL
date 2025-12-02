import pandas as pd
df = pd.read_csv('results/training_log.csv')
df['reward_moving_avg'] = df['reward'].rolling(window=100).mean()
print(df[['episode', 'reward', 'reward_moving_avg']].tail(10))  # Last example
