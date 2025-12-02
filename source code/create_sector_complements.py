import pandas as pd
import numpy as np
import pathlib
import os
import random

# List of department codes (fixed, based on standard WIOD)
SECTOR_CODES = [
    'A01', 'A02', 'A03', 'B', 'C10-C12', 'C13-C15', 'C16', 'C17', 'C18', 'C19',
    'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26', 'C27', 'C28', 'C29',
    'C30', 'C31_C32', 'C33', 'D35', 'E36', 'E37-E39', 'F', 'G45', 'G46', 'G47',
    'H49', 'H50', 'H51', 'H52', 'H53', 'I', 'J58', 'J59_J60', 'J61', 'J62_J63',
    'K64', 'K65', 'K66', 'L68', 'M69_M70', 'M71', 'M72', 'M73', 'M74_M75', 'N',
    'O84', 'P85', 'Q', 'R_S', 'T', 'U'
]

# Country to offset map (for 224 nodes: 56 sectors × 4 countries)
NODE_MAP = {'CHN': 0, 'USA': 56, 'TWN': 112, 'MEX': 168}


# Default year for calculating correlation
def create_sector_complements(year=2014):
    """
   Generating complementary pairs of sectors based on Spearman correlation and specific economic pairs.
   The output is saved in the file data/sector_complements.py.
    """
    path = pathlib.Path(f"data/processed/WIOT_{year}_filtered_with_codes.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"file {path} Not found! Run create_country_matrix.py first.")

    # Load filtered data
    df = pd.read_csv(path, header=[0, 1], index_col=[0, 1])
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Calculating Spearman correlation on the filtered matrix
    corr_matrix = df.corr(method='spearman')
    np.fill_diagonal(corr_matrix.values, 0)  # Zeroing the main diameter

    # Select top_k pairs with the highest correlation
    top_k = 10
    flat_indices = np.argsort(corr_matrix.values.ravel())[-top_k:]
    row_idx, col_idx = np.unravel_index(flat_indices, corr_matrix.shape)
    general_complements = [[row_idx[i], col_idx[i]]
                           for i in range(len(row_idx))]

    # Specific economic pairs (US-MEX, CHN-TWN)
    us_mex_pairs = []
    chn_twn_pairs = []
    for i, us_sector in enumerate(SECTOR_CODES):
        for j, mex_sector in enumerate(SECTOR_CODES):
            us_node = NODE_MAP['USA'] + i
            mex_node = NODE_MAP['MEX'] + j
            us_mex_pairs.append([us_node, mex_node])
    for i, chn_sector in enumerate(SECTOR_CODES):
        for j, twn_sector in enumerate(SECTOR_CODES):
            chn_node = NODE_MAP['CHN'] + i
            twn_node = NODE_MAP['TWN'] + j
            chn_twn_pairs.append([chn_node, twn_node])

    # Randomly select 10 pairs of each
    economic_complements = random.sample(us_mex_pairs, min(10, len(us_mex_pairs))) + \
        random.sample(chn_twn_pairs, min(10, len(chn_twn_pairs)))

    # Combination of general and economic pairs
    complements = general_complements + economic_complements

    # Check indexes (must be <224)
    if any(x >= 224 for pair in complements for x in pair):
        raise ValueError("index >= 224 Found! Check the sections.")

   # Generate output file
    module_content = f"""
# Auto-generated sector complement pairs for US-China Shock (filtered data)
SECTOR_COMPLEMENTS = {complements}

def get_complement_pairs(n=10, sector_codes=None):
    \"\"\"
   Randomly select n complementary pairs.
   If sector_codes is given, it can filter pairs (currently no filter).
    \"\"\"
    return random.choices(SECTOR_COMPLEMENTS, k=n)
"""
    os.makedirs("data", exist_ok=True)
    output_path = "data/sector_complements.py"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(module_content)

    print(f"file {output_path} It was made.")
    print(f"  Number of complementary pairs: {len(complements)}")


if __name__ == "__main__":
    create_sector_complements()
