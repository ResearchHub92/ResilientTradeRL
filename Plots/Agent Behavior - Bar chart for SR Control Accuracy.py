import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('results/training_log.csv')
accuracy = (abs(df['sr'] - 0.5) < 0.05).mean() * 100  # Sample interval
plt.bar(['Accuracy'], [accuracy])
plt.ylabel('Percentage')
plt.title('SR Control Accuracy Bar Chart')
plt.savefig('sr_control_accuracy_bar.png')
plt.close()
