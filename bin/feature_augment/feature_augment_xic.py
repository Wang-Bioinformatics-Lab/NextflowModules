import os
import sys
import argparse
import pandas as pd
from scipy import integrate
from massql import msql_fileloading

def _perform_xic(mzml_filename, target_list_df, mz_da_tolernace=0.01, rt_min_tolernace=0.1):
    ms1_df, ms2_df = msql_fileloading.load_data(mzml_filename)

    # lets do something now
    target_list = target_list_df.to_dict(orient='records')

    # TODO: Make sure we list all the MS1 scans with 0 intensity if mz is not found
    empty_df = ms1_df.groupby('scan').first()
    empty_df["scan"] = empty_df.index
    empty_df["i"] = 0

    output_list = []

    for target_obj in target_list:
        max_mz = target_obj['row m/z'] + mz_da_tolernace
        min_mz = target_obj['row m/z'] - mz_da_tolernace

        max_rt = target_obj['row retention time'] + rt_min_tolernace
        min_rt = target_obj['row retention time'] - rt_min_tolernace

        # Filtering the ms1_df 
        filtered_ms1_df = ms1_df[(ms1_df['mz'] >= min_mz) & (ms1_df['mz'] <= max_mz) & (ms1_df['rt'] >= min_rt) & (ms1_df['rt'] <= max_rt)]

        if len(filtered_ms1_df) == 0:
            calculated_area = 0
        else:
            # grouping by the scan
            grouped_ms1_df = filtered_ms1_df.groupby('scan')

            # summing the intensity
            summed_intensity_df = pd.DataFrame()
            summed_intensity_df["i"] = grouped_ms1_df['i'].sum()
            summed_intensity_df["scan"] = grouped_ms1_df['scan'].first()
            summed_intensity_df["rt"] = grouped_ms1_df['rt'].first()

            included_scans = set(summed_intensity_df['scan'])
            missing_scans = set(empty_df['scan']) - included_scans

            filtered_empty_df = empty_df[empty_df['scan'].isin(missing_scans)]

            summed_intensity_df = pd.concat([summed_intensity_df, filtered_empty_df])

            # sorting the df
            summed_intensity_df = summed_intensity_df.sort_values(by='rt')

            # doing the trapz
            calculated_area = integrate.trapezoid(summed_intensity_df["i"], x=summed_intensity_df["rt"])

        output_dict = {}
        output_dict['row ID'] = target_obj['row ID']
        output_dict['row m/z'] = target_obj['row m/z']
        output_dict['row retention time'] = target_obj['row retention time']
        output_dict["area"] = calculated_area
        output_dict["filename"] = os.path.basename(mzml_filename)

        output_list.append(output_dict)

    return pd.DataFrame(output_list)

def main():
    parser = argparse.ArgumentParser(description='Test write out a file.')
    parser.add_argument('input_mzml_filename')
    parser.add_argument('input_target_list')
    parser.add_argument('output_quant_filename')

    parser.add_argument('--mz_da_tolernace', default=0.01, type=float, help='The m/z tolerance for the XIC')
    parser.add_argument('--rt_min_tolernace', default=0.1, type=float, help='The retention time tolerance for the XIC in minutes')

    args = parser.parse_args()

    print(args)

    # reading the targetlist
    target_list_df = pd.read_csv(args.input_target_list, sep=',')

    quant_df = _perform_xic(args.input_mzml_filename, target_list_df, mz_da_tolernace=args.mz_da_tolernace, rt_min_tolernace=args.rt_min_tolernace)

    # outputting
    quant_df.to_csv(args.output_quant_filename, index=False, sep=",")



if __name__ == "__main__":
    main()