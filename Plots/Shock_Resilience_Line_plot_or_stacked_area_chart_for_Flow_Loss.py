import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('results/shock_results_filtered.csv')
plt.plot(df['Year'], df['Flow_Loss'], marker='o')
plt.xlabel('Year')
plt.ylabel('Flow Loss')
plt.title('Flow Loss Over Years')
plt.savefig('flow_loss_line.png')
plt.close()