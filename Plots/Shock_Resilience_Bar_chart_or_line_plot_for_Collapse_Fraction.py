import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('results/shock_results_filtered.csv')
plt.bar(df['Year'], df['Collapse_Fraction'])
plt.xlabel('Year')
plt.ylabel('Collapse Fraction')
plt.title('Collapse Fraction by Year')
plt.savefig('collapse_fraction_bar.png')
plt.close()