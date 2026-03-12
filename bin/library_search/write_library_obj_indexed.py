import sys
import time
import argparse
import numpy as np
import pickle

import gnps_index
from library_search_indexed import read_file, convert_to_mamba_format

def main():
    parser = argparse.ArgumentParser(description='Running library search parallel')
    parser.add_argument('library_file', help='library_file')
    parser.add_argument('result_folder', help='output folder for results')
    parser.add_argument('convert_binary', help='conversion binary')
    parser.add_argument('librarysearch_binary', help='librarysearch_binary')

    parser.add_argument('--fragment_tolerance', default=0.5, help='fragment_tolerance', type=float)
    parser.add_argument('--threads', default=1, type=int, help='Number of threads to use for parallel processing')
    parser.add_argument('--filter_precursor', default=1, type=int, help='Filter precursor peaks (1) or not (0)')
    parser.add_argument('--filter_window', default=1, type=int, help='Filter peaks in a window (1) or not (0)')

    args = parser.parse_args()
    # start_time = time.time()
    # with open("library_obj.pkl", "rb") as f:
    #     library_obj = pickle.load(f)
    # print("Reading the pickle took:", time.time() - start_time, "seconds")
    # sys.exit(0)

    try:
        print("starting to read files", args.library_file)
        start_time = time.time()
        library_mgf = read_file(args.library_file)
        print("Reading the files took:", time.time() - start_time, "seconds")
        start_time = time.time()
    except Exception as e:
        print("Not able to read the input files. Exiting.", e)
        sys.exit(0)
        
    if len(library_mgf) == 0:
        print("No spectra found in the input files. Exiting.")
        sys.exit(0)
    
    def convert_single_spectrum_to_spectrum_list(data_dict):
        mz = [float(x[0]) for x in data_dict['PEAKS']]
        intensity = [float(x[1]) for x in data_dict['PEAKS']]
        precursor_mz = np.float32(float(data_dict.get('PEPMASS', 0.0))) # precursor mass is converted to float 32 to match legacy GNPS result.
        precursor_charge = data_dict.get('CHARGE', data_dict.get('charge', 1))
        precursor_charge = abs(int(precursor_charge))  # Ensure charge is positive
        mz = np.asarray(mz)
        intensity = np.asarray(intensity)
        return gnps_index.filter_peaks_optimized(mz, intensity, precursor_mz, precursor_charge,
                                                 filter_precursor=(args.filter_precursor == 1),
                                                 filter_window=(args.filter_window == 1))

    library_list = [convert_single_spectrum_to_spectrum_list(spec) for spec in library_mgf]
    print("Converting spectra to spectrum list format and cleaning took:", time.time() - start_time, "seconds")
    start_time = time.time()
    
    # converted_library_list, num_library = convert_to_mamba_format(library_list)
    # print("Converting spectra to mamba format took:", time.time() - start_time, "seconds")
    # start_time = time.time()
    
    # library_shared_idx = gnps_index.create_index(converted_library_list, False, args.fragment_tolerance, gnps_index.SHIFTED_OFFSET)
    # library_shifted_idx = gnps_index.create_index(converted_library_list, True, args.fragment_tolerance, gnps_index.SHIFTED_OFFSET)

    # save
    library_obj = dict(
        library_mgf=library_mgf,
        library_list=library_list,
        # library_shared_idx=library_shared_idx,
        # library_shifted_idx=library_shifted_idx,
    )
    with open("library_obj.pkl", "wb") as f:
        pickle.dump(library_obj, f)

if __name__ == "__main__":
    main()