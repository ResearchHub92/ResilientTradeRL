from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('results/shock_results_filtered.csv')
t_stat, p_val = ttest_ind(df['Before_SR'], df['After_SR'])
bars = plt.bar(['Before', 'After'], [
               df['Before_SR'].mean(), df['After_SR'].mean()])
if p_val < 0.05:
    plt.text(1, df['After_SR'].mean(), '*', ha='center')
plt.ylabel('Mean SR')
plt.title('t-test Comparison Bar Chart')
plt.savefig('ttest_bar.png')
plt.close()
# Explanation: For Wilcoxon, use wilcoxon if the data is paired.
