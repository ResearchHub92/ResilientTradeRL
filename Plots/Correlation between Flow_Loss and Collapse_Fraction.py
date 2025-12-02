import pandas as pd
from scipy.stats import pearsonr

df = pd.read_csv('results/shock_results_filtered.csv')
corr, p = pearsonr(df['Flow_Loss'], df['Collapse_Fraction'])
print(f'Correlation: {corr:.2f}, p-value: {p:.2f}')
# Explanation: For the article, add this statistic
