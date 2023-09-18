import os
import sys
import pandas as pd

if __name__ == '__main__':
    input_file = sys.argv[1]

    if len(sys.argv) > 2:
        batch_size = int(sys.argv[2])
    else:
        batch_size = 50

    # Read the csv file into a dataframe
    # then split the dataframe into multiple dataframes.
    # Each dataframe will have batch_size samples and be saved as a separated csv file.
    df = pd.read_csv(input_file)
    df_list = [df.loc[i:i+batch_size-1, :]
               for i in range(0, len(df), batch_size)]

    # Save the dataframes as csv files
    df.to_excel(input_file[:-4] + '.xlsx', index=False)
    for i, df in enumerate(df_list):
        df.to_excel('Splitted/' + os.path.basename(input_file)
                    [:-4] + '_{}.xlsx'.format(i), index=False)
