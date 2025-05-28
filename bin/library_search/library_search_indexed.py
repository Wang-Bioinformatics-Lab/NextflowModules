import sys
import getopt
import os
import json
import argparse
import uuid
import pandas as pd
from collections import defaultdict

import gnps_index
from numba.typed import List
import numpy as np
from pyteomics import mgf
import numba

# BUFFER_SIZE = 2000 

def convert_to_mamba_format(file_list):
    numba_spectra = List()
    for spec in file_list:
        numba_spectra.append((
            spec[0].astype(np.float32),
            spec[1].astype(np.float32),
            np.float32(spec[2]),
            np.int32(spec[3])
        ))
    n_spectra = len(numba_spectra)
    return numba_spectra, n_spectra

@numba.njit
def cross_library_compute_all_pairs(spectra, library, shared_entries,
                                    shifted_entries, tolerance,
                                    threshold, topk,
                                    library_min_matched_peaks):
    results = List()

    n_library = len(library)
    n_spectra = len(spectra)
    for query_idx in range(n_spectra):
        # print(f"Processing query spectrum {query_idx + 1}/{n_spectra}...")
        query_spec = spectra[query_idx]
        upper_bounds = np.zeros(n_library, dtype=np.float32)
        match_counts = np.zeros(n_library, dtype=np.int32)

        # Process both shared and shifted peaks
        for peak_idx in range(len(query_spec[0])):
            mz = query_spec[0][peak_idx]
            intensity = query_spec[1][peak_idx]
            precursor_mz = query_spec[2]

            # Shared peaks processing
            shared_bin = np.int64(round(mz / tolerance))
            shifted_bin = np.int64(round((precursor_mz - mz + gnps_index.SHIFTED_OFFSET) / tolerance))

            # Check both shared and shifted entries
            for entries, bin_val in [(shared_entries, shared_bin),
                                     (shifted_entries, shifted_bin)]:
                for delta in gnps_index.ADJACENT_BINS:
                    target_bin = bin_val + delta
                    start, end = gnps_index.find_bin_range(entries, target_bin)
                    pos = start
                    while pos < end and entries[pos][1] <= query_idx:
                        pos += 1
                    # Find matches in this bin
                    while pos < end and entries[pos][0] == target_bin:
                        spec_idx = entries[pos][1]
                        upper_bounds[spec_idx] += intensity * entries[pos][4]
                        match_counts[spec_idx] += 1
                        pos += 1
        # print("log", f"Upper bounds for query {query_idx}: {len(upper_bounds)}")
        # Collect candidates using threshold parameter
        # print("log", f"Collecting candidates for query {query_idx} where total matches >= {library_min_matched_peaks} and n_library = {n_library}... with threshold", threshold)
        candidates = List()
        for lib_idx in range(n_library):
            # if query_idx == 41 and lib_idx == 22511:
                # print("log", "Processing query 41, library index 22511...", upper_bounds[lib_idx], match_counts[lib_idx], upper_bounds[lib_idx] >= threshold, match_counts[lib_idx] >= library_min_matched_peaks)
            if (upper_bounds[lib_idx] >= threshold and
                    match_counts[lib_idx] >= library_min_matched_peaks):
                candidates.append((lib_idx, upper_bounds[lib_idx]))

        # print("log", f"Candidates for query {query_idx}: {len(candidates)}")
        # for item in candidates:
        #     print("log", f"Candidate: {item[0]}, Score: {item[1]}")
        # Sort candidates by upper bound score
        candidates.sort(key=lambda x: -x[1])
        # print("log", f"sorted candidates for query {query_idx}: {len(candidates)}")

        # Process top candidates for exact matching
        exact_matches = List()
        for lib_idx, _ in candidates:
            # print("log", f"Processing exact match for candidate {lib_idx}...")
            target_spec = library[lib_idx]
            score, shared, shifted, num_matches = gnps_index.calculate_exact_score_GNPS(spectra[query_idx], target_spec, tolerance)
            # print("log", f"Score for candidate {lib_idx}: {score}, Shared: {shared}, Shifted: {shifted}, Matches: {num_matches}")
            if score >= threshold:
                exact_matches.append((lib_idx, score, shared, shifted, num_matches))

        # Sort and store top results
        exact_matches.sort(key=lambda x: -x[1])
        size_val = min(topk, len(exact_matches))
        results.append((query_idx, exact_matches[:size_val]))

    return results

def convert_to_legacy_format(matches, spectra, library, topk, min_cosine, min_matched_peaks, qry_file_name):
    matches_buffer = []
    for query_idx, candidates in matches:
        q_spec = spectra[query_idx]
        for cand in candidates:
            matches_buffer.append({
            '#Scan#': q_spec["SCANS"],
            'SpectrumFile': qry_file_name,
            'Charge': library[cand[0]].get('CHARGE', 1),
            'MQScore': round(cand[1], 4),
            # 'p-value': qry_spec.rt,  # RT value # this needs to be fixed in the future
            # 'UnstrictEvelopeScore': round(qry_spec.tic),  # TIC value # this needs to be fixed in the future
            'FileScanUniqueID': f'{qry_file_name}_{q_spec["SCANS"]}',
            'mzErrorPPM': round((q_spec["PEPMASS"] - library[cand[0]]["PEPMASS"]) / library[cand[0]]["PEPMASS"] * 1e6, 4),
            'LibSearchSharedPeaks': cand[4],
            'ParentMassDiff': round(q_spec["PEPMASS"] - library[cand[0]]["PEPMASS"], 4),
            'SpecMZ': q_spec["PEPMASS"],
            'LibrarySpectrumID': library[cand[0]]['SPECTRUMID'] if library[cand[0]]['SPECTRUMID'] != '' else f'scans_{library[cand[0]]["SCANS"]}',
            'Smiles': library[cand[0]].get('SMILES', ''),
        })
    
    print("log", f"Converted {len(matches_buffer)} matches to legacy format.")
    
    results_df = pd.DataFrame(matches_buffer)

    columns = ['#Scan#', 'SpectrumFile', 'Annotation', 'OrigAnnotation', 'Protein', 'dbIndex', 'numMods', 'matchOrientation', 'startMass', 'Charge', 'MQScore', 'p-value', 'isDecoy', 'StrictEnvelopeScore', 'UnstrictEvelopeScore', 'CompoundName', 'Organism', 'FileScanUniqueID', 'FDR', 'LibraryName', 'mzErrorPPM', 'LibMetaData', 'Smiles', 'Inchi', 'LibSearchSharedPeaks', 'Abundance', 'ParentMassDiff', 'SpecMZ', 'ExactMass', 'LibrarySpectrumID']
    # for each column in columns, if it is not in results_df, add it with empty values
    for col in columns:
        if col not in results_df.columns:
            results_df[col] = ''
    return results_df
    
    

def read_mgf_spectrum(file_obj):
    """Read a single spectrum block from an open MGF file.
    Args:
        file_obj: An opened file object positioned at the start of a spectrum
    Returns:
        dict: Spectrum information containing all metadata and peaks,
        or None if end of file is reached
    """
    # Initialize spectrum with common metadata fields
    spectrum = {
        'PEPMASS': 0.0,
        'SPECTRUMID': '',
        'SCANS': '',
        'PEAKS': []
    }

    # Skip any empty lines before BEGIN IONS
    for line in file_obj:
        if line.strip() == 'BEGIN IONS':
            break
    else:  # EOF reached
        return None

    # Read spectrum metadata and peaks
    for line in file_obj:
        line = line.strip()

        if not line:  # Skip empty lines
            continue

        if line == 'END IONS':
            if spectrum['PEAKS']:
                this_peaks = np.asarray(spectrum['PEAKS'])
                # this_peaks[:, 1] = this_peaks[:, 1] / np.max(this_peaks[:, 1]) * 999
                this_peaks = this_peaks[np.bitwise_and(this_peaks[:, 0] > 0, this_peaks[:, 1] > 0)]
                spectrum['PEAKS'] = np.asarray(this_peaks, dtype=np.float32)

            # new_spectrum = {key.tolower(): value for key, value in spectrum.items()}
            return spectrum

        # Handle peak data
        if line and not line.startswith(('BEGIN', 'END')):
            if '=' in line:
                # First try to parse as metadata
                key, value = line.split('=', 1)

                # Handle specific numeric fields
                if key in ['PEPMASS', 'PRECURSORMZ']:
                    try:
                        spectrum[key] = float(value.strip())
                    except:
                        continue
                else:
                    if key in ['CHARGE', 'charge']:
                        if type(value) is str:
                            sign = 1
                            if "-" in value:
                                value = value.replace("-", "")
                                sign = -1
                            value = value.replace("+", "")
                            if value.isdigit():
                                value=int(value) * sign
                            else:
                                value=1* sign  # Default to 1 if not a valid integer
                        else:
                            value = int(value)
                    
                    key = key.strip().upper()
                    spectrum[key] = value
            else:
                # If no '=' found, treat as peak data
                try:
                    mz, intensity = line.split()
                    spectrum['PEAKS'].append((float(mz), float(intensity)))
                except:
                    continue

    return None


def read_mgf(mgf_path, buffer_size=8192):
    """Iterate through spectra in an MGF file efficiently using buffered reading.
    Args:
        mgf_path: Path to the MGF file
        buffer_size: Read buffer size in bytes
    Returns:
        list: List of spectra, each represented as a dictionary
    """
    result = []
    with open(mgf_path, 'r', buffering=buffer_size) as f:
        while True:
            spectrum = read_mgf_spectrum(f)
            if spectrum is None:
                break
            result.append(spectrum)
        
    return result

def main():
    parser = argparse.ArgumentParser(description='Running library search parallel')
    parser.add_argument('spectrum_file', help='spectrum_file')
    parser.add_argument('library_file', help='library_file')
    parser.add_argument('result_folder', help='output folder for results')
    parser.add_argument('convert_binary', help='conversion binary')
    parser.add_argument('librarysearch_binary', help='librarysearch_binary')

    parser.add_argument('--pm_tolerance', default=0.5, help='pm_tolerance', type=float)
    parser.add_argument('--fragment_tolerance', default=0.5, help='fragment_tolerance', type=float)
    parser.add_argument('--library_min_cosine', default=0.7, help='library_min_cosine', type=float)
    parser.add_argument('--library_min_matched_peaks', default=6, help='library_min_matched_peaks', type=int)
    parser.add_argument('--topk', default=1, help='topk', type=int)
    parser.add_argument('--analog_search', default=0, help='Turn on analog search, 0 or 1', type=int)
    parser.add_argument('--threads', default=1, type=int, help='Number of threads to use for parallel processing')

    # This is good for bookkeeping in GNPS2 if you need the full path
    parser.add_argument('--full_relative_query_path', default=None, help='This is the original full relative path of the input file')
    
    args = parser.parse_args()
    # print("topk", args.topk)
    # print("spectrum_file:", args.spectrum_file)
    
    spectrum_mgf = read_mgf(args.spectrum_file)
    # print(spectrum_mgf[0].keys())

    library_mgf = read_mgf(args.library_file)
    
    def convert_single_spectrum_to_spectrum_list(data_dict):
        mz = [float(x[0]) for x in data_dict['PEAKS']]
        intensity = [float(x[1]) for x in data_dict['PEAKS']]
        precursor_mz = float(data_dict.get('PEPMASS', 0.0))
        precursor_charge = data_dict.get('CHARGE', data_dict.get('charge', 1))
        precursor_charge = abs(int(precursor_charge))  # Ensure charge is positive
        # print("log", f"Converted spectrum with {len(mz)} peaks, precursor_mz: {precursor_mz}, precursor_charge: {precursor_charge}")
        # print("log", f"mz: {mz[:5]}, intensity: {intensity[:5]}")
        mz = np.asarray(mz)
        intensity = np.asarray(intensity)
        return gnps_index.filter_peaks_optimized(mz, intensity, precursor_mz, precursor_charge)
    
    spectrum_list = [convert_single_spectrum_to_spectrum_list(spec) for spec in spectrum_mgf]
    library_list = [convert_single_spectrum_to_spectrum_list(spec) for spec in library_mgf]
    
    print("log", f"Converted {len(spectrum_list)} spectra from query file and {len(library_list)} spectra from library file.")
    
    converted_spectrum_list, num_spectra = convert_to_mamba_format(spectrum_list)
    converted_library_list, num_library = convert_to_mamba_format(library_list)
    
    library_shared_idx = gnps_index.create_index(converted_library_list, False, args.fragment_tolerance, gnps_index.SHIFTED_OFFSET)
    library_shifted_idx = gnps_index.create_index(converted_library_list, True, args.fragment_tolerance, gnps_index.SHIFTED_OFFSET)
    
    matches = cross_library_compute_all_pairs(converted_spectrum_list,
                                              converted_library_list,
                                              library_shared_idx,
                                              library_shifted_idx,
                                args.fragment_tolerance, args.library_min_cosine, args.topk, args.library_min_matched_peaks)

    print("log", f"Found {len(matches)} matches across {num_spectra} query spectra and {num_library} library spectra.")
    # print(library_mgf[0].keys())
    results_df = convert_to_legacy_format(matches, spectrum_mgf, library_mgf, args.topk, args.library_min_cosine, args.library_min_matched_peaks, args.spectrum_file)
    # sort results_df by MQScore
    results_df = results_df.sort_values(by='MQScore', ascending=False)
    print("log", f"Converted results to DataFrame with {len(results_df)} rows.")
    
    results_df["SpectrumFile"] = results_df["SpectrumFile"].apply(lambda x: os.path.basename(x))

    if not os.path.exists(args.result_folder):
        os.makedirs(args.result_folder)
    
    # We should rewrite if we have the full path
    if args.full_relative_query_path is not None:
        results_df["SpectrumFile"] = args.full_relative_query_path

        # Create a safe filename from a full path using from args.full_relative_query_path
        safe_filename = args.full_relative_query_path.replace("/", "_").replace(".", "_").replace(" ", "_")

        # lets hash safe_filename to make it shorter
        safe_filename = str(uuid.uuid3(uuid.NAMESPACE_DNS, safe_filename)) + ":" + safe_filename[-20:]

        # fixing the output filename
        output_results_file = os.path.join(args.result_folder, safe_filename + "_" + os.path.basename(args.library_file) + ".tsv")

    results_df.to_csv(output_results_file, sep="\t", index=False)
    
if __name__ == "__main__":
    main()