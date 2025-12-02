import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('results/shock_results_filtered.csv')
plt.boxplot(df['After_SR'])
plt.xticks([1], ['After_SR Across Years'])
plt.ylabel('SR Value')
plt.title('Mean ± Std of SR Boxplot')
plt.savefig('mean_std_sr_boxplot.png')
plt.close()