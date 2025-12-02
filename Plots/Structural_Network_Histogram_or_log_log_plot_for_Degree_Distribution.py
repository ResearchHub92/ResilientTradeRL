# import matplotlib.pyplot as plt
# import ast
# import pandas as pd
# import numpy as np

# df = pd.read_csv('results/shock_results_filtered.csv')
# degrees = ast.literal_eval(df['Degree_Dist'][0])  # year 2000

# plt.hist(degrees, bins=20, log=True)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Degree (log scale)')
# plt.ylabel('Frequency (log scale)')
# plt.title('Degree Distribution (Log-Log Plot)')
# plt.savefig('degree_distribution.png')
# plt.close()
# # Explanation: Loop over df for all years.

import matplotlib.pyplot as plt
import ast
import pandas as pd
import numpy as np

df = pd.read_csv('results/shock_results_filtered.csv')

for index, row in df.iterrows():
    year = row['Year']
    degrees = ast.literal_eval(row['Degree_Dist'])

    # Remove zero values ​​for logarithmic graph
    non_zero_degrees = [d for d in degrees if d > 0]

    plt.figure(figsize=(10, 6))
    # alpha=0.7, color='skyblue')
    plt.hist(non_zero_degrees, bins=20, log=True)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Degree (log scale)')
    plt.ylabel('Frequency (log scale)')
    plt.title(f'Degree Distribution (Log-Log Plot{year})')
    plt.grid(True, alpha=0.3)

    plt.savefig(f'degree_distribution_{year}.png', dpi=300)
    plt.close()
