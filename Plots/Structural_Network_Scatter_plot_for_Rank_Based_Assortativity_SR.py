import matplotlib.pyplot as plt
import ast
import pandas as pd

df = pd.read_csv('results/shock_results_filtered.csv')  # With new columns
in_deg = ast.literal_eval(df['In_Degree'][0])  # Sample for the year 2000
out_deg = ast.literal_eval(df['Out_Degree'][0])

plt.scatter(in_deg, out_deg, alpha=0.5)
plt.xlabel('In-Degree')
plt.ylabel('Out-Degree')
plt.title('Scatter Plot for Rank-Based Assortativity (SR)')
plt.savefig('sr_scatter.png')
plt.close()
# Explanation: You can add color schemes based on country.
