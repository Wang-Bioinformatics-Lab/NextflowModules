import pandas as pd
import argparse
import glob

def main():
    parser = argparse.ArgumentParser(description='reformatting')
    parser.add_argument('quant_file')
    parser.add_argument('output_quant_filename')

    args = parser.parse_args()

    quant_df = pd.read_csv(args.quant_file, sep=",")

    # we are going to reformat the quant file, by making it a wide table
    wide_df = pd.pivot(quant_df, index='row ID', columns='filename', values='area')

    # rename columns to add peak area, excluding the columns that have the "row" prefix
    wide_df.columns = [col + " Peak area" if not col.startswith('row') else col for col in wide_df.columns]
    
    # adding in the row m/z and retention time by grouping the quant_df
    row_df = quant_df.groupby('row ID').first()
    row_df = row_df[['row m/z', 'row retention time']]
    
    # merging in
    wide_df = wide_df.merge(row_df, left_on="row ID", right_index=True)

    wide_df.to_csv(args.output_quant_filename, sep=",")


if __name__ == "__main__":
    main()