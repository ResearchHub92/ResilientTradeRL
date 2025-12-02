import matplotlib.pyplot as plt
from scipy.stats import linregress
import ast
import pandas as pd
import numpy as np

df = pd.read_csv('results/shock_results_filtered.csv')  # With new columns
in_deg = ast.literal_eval(df['In_Degree'][0])
out_deg = ast.literal_eval(df['Out_Degree'][0])

slope, intercept, r_value, p_value, std_err = linregress(in_deg, out_deg)
plt.scatter(in_deg, out_deg, alpha=0.5)
plt.plot(in_deg, slope * np.array(in_deg) + intercept, 'r--')
plt.xlabel('In-Degree')
plt.ylabel('Out-Degree')
plt.title('Pearson Assortativity Scatter with Linear Regression')
plt.text(0.05, 0.95, f'r = {r_value:.2f}', transform=plt.gca().transAxes)
plt.savefig('pearson_scatter.png')
plt.close()
