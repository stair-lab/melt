import os
import sys

import pandas as pd


if __name__ == "__main__":
    fname = sys.argv[1]
    num_of_files = int(sys.argv[2])

    # Merge all csv files into one
    df = pd.read_csv(fname + "_0.csv")
    for i in range(1, num_of_files):
        print(f"Processing {fname}_{i}.csv")
        new_df = pd.read_csv(fname + f"_{i}.csv")
        df = pd.concat([df, new_df], ignore_index=True)

    # Save the merged file
    df.to_csv(fname + ".csv", index=False)
