import pandas as pd
import numpy as np
import os

# setting
YEARS = range(2000, 2015)
TARGET_COUNTRIES = ['CHN', 'USA', 'TWN', 'MEX']
OUTPUT_DIR = "data/processed"

# Main sections (without final demand)
MAIN_SECTORS = [
    'A01', 'A02', 'A03', 'B', 'C10-C12', 'C13-C15', 'C16', 'C17', 'C18', 'C19',
    'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26', 'C27', 'C28', 'C29',
    'C30', 'C31_C32', 'C33', 'D35', 'E36', 'E37-E39', 'F', 'G45', 'G46', 'G47',
    'H49', 'H50', 'H51', 'H52', 'H53', 'I', 'J58', 'J59_J60', 'J61', 'J62_J63',
    'K64', 'K65', 'K66', 'L68', 'M69_M70', 'M71', 'M72', 'M73', 'M74_M75', 'N',
    'O84', 'P85', 'Q', 'R_S', 'T', 'U'
]

# Final demand sections that should be removed
FINAL_DEMAND_SECTORS = ['CONS_h', 'CONS_np', 'CONS_g', 'GFCF', 'INVEN']

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_and_filter_wiod(year):
    path = f"data/wiod2016/WIOT{year}_Nov16_ROW.xlsb"
    if not os.path.exists(path):
        raise FileNotFoundError(f"file {path} Not found!")

    try:
        #  Load file
        df = pd.read_excel(path, sheet_name=0, engine="pyxlsb",
                           header=[2, 4], index_col=[0, 2])

        # Delete rows and columns TOTAL
        df = df.drop(columns=["TOTAL"], level=0, errors='ignore')
        df = df.drop(index=["TOTAL"], level=0, errors='ignore')

        # Clean up the index and columns
        def clean_index(idx):
            if isinstance(idx, tuple):
                return tuple(str(x).strip() if pd.notna(x) else 'UNKNOWN' for x in idx)
            else:
                return (str(idx).strip(), 'UNKNOWN')

       # Apply cleanup to the index and columns
        df.index = pd.MultiIndex.from_tuples(
            [clean_index(idx) for idx in df.index])
        df.columns = pd.MultiIndex.from_tuples(
            [clean_index(col) for col in df.columns])

        print(f"year {year}: Main dimensions {df.shape}")

        # Filter only the desired countries and remove final demand
        country_cols = [col for col in df.columns
                        if col[1] in TARGET_COUNTRIES
                        and col[0] not in FINAL_DEMAND_SECTORS]

        country_rows = [idx for idx in df.index
                        if idx[1] in TARGET_COUNTRIES
                        and idx[0] not in FINAL_DEMAND_SECTORS]

        if len(country_cols) == 0 or len(country_rows) == 0:
            print(f"year {year}: No country of {TARGET_COUNTRIES} Not found!")
            return None

        filtered_df = df.loc[country_rows, country_cols]

        print(f"year {year}: Filtered dimensions {filtered_df.shape}")
        print(
            f"year {year}: Number of non-zero values {(filtered_df != 0).sum().sum()}")

        # Save the filtered matrix
        output_path = os.path.join(
            OUTPUT_DIR, f"WIOT_{year}_filtered_with_codes.csv")
        filtered_df.to_csv(output_path, index=True)
        print(f"year {year}: The filtered matrix was saved in {output_path}")

        return filtered_df

    except Exception as e:
        print(f"  Error processing the year  {year}: {e}")
        return None


# Processing all years
for year in YEARS:
    load_and_filter_wiod(year)
