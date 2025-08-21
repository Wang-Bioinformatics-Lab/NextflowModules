import os
import sys
import pandas as pd
import argparse
import requests
import requests_cache
from tqdm import tqdm
from pyteomics import mgf
from pyteomics import mzml
from gnpsdata import fasst

# this provides a list of queries dicts
def prep_query_mgf_queries(querymgf_filename, database, analog=False, precursor_mz_tol=0.02, fragment_mz_tol=0.02, min_cos=0.7):
    mgf_data = mgf.MGF(querymgf_filename)
    queries = []

    for spectrum in mgf_data:
        mz = spectrum["m/z array"]
        intensity = spectrum["intensity array"]
        peaks_zip = list(zip(mz, intensity))
        precursor_mz = spectrum["params"]["pepmass"][0]

        query_scan = spectrum["params"]["scans"]

        query = {
            "datasource": "mgf",
            "precursor_mz": precursor_mz,
            "peaks": peaks_zip,
            "database": database,
            "analog": analog,
            "precursor_mz_tol": precursor_mz_tol,
            "fragment_mz_tol": fragment_mz_tol,
            "min_cos": min_cos,
            "query_scan": query_scan
        }

        queries.append(query)

    return queries

def execute_all_queries(queries):
    output_results_list = []

    for query in tqdm(queries):
        try:
            results_dict = fasst.query_fasst_api_peaks(
                query["precursor_mz"],
                query["peaks"],
                query["database"],
                analog=query["analog"],
                precursor_mz_tol=query["precursor_mz_tol"],
                fragment_mz_tol=query["fragment_mz_tol"],
                min_cos=query["min_cos"]
            )
            
            if "status" in results_dict:
                    #print(results_dict["status"])
                    #print("Error in Results.")
                    continue

            results_df = pd.DataFrame(results_dict["results"])

            # adding the scan number information
            results_df["query_scan"] = query["query_scan"]

            output_results_list.append(results_df)
        except KeyboardInterrupt:
            raise
        except:
            print("Error in Query")
            #raise
            pass

    if len(output_results_list) == 0:
        print("No results found")
        return pd.DataFrame()

    output_results_df = pd.concat(output_results_list)

    return output_results_df






















def masst_query_mgf_all(querymgf_filename, database, analog=False, precursor_mz_tol=0.02, fragment_mz_tol=0.02, min_cos=0.7):
    # Read the MGF file
    mgf_data = mgf.MGF(querymgf_filename)
    output_results_list = []

    for spectrum in tqdm(mgf_data):
        try:
            #print(spectrum)
            mz = spectrum["m/z array"]
            intensity = spectrum["intensity array"]

            peaks_zip = list(zip(mz, intensity))

            precursor_mz = spectrum["params"]["pepmass"][0]

            results_dict = fasst.query_fasst_api_peaks(precursor_mz, peaks_zip, database,
                analog=analog, 
                precursor_mz_tol=precursor_mz_tol, 
                fragment_mz_tol=fragment_mz_tol, 
                min_cos=min_cos)

            #print(results_dict.keys())
            
            if "status" in results_dict:
                #print(results_dict["status"])
                #print("Error in Results.")
                continue

            results_df = pd.DataFrame(results_dict["results"])

            # adding the scan number information
            results_df["query_scan"] = spectrum["params"]["scans"]

            output_results_list.append(results_df)
        except KeyboardInterrupt:
            raise
        except:
            print("Error in Query")
            #raise
            pass

    if len(output_results_list) == 0:
        print("No results found")
        return pd.DataFrame()

    output_results_df = pd.concat(output_results_list)

    return output_results_df

def _get_scan(spectrum_id_string):
    if "scan" in spectrum_id_string:
        return spectrum_id_string.split("scan=")[-1]

def masst_query_mzml_all(query_mzml_filename, database, analog=False, precursor_mz_tol=0.02, fragment_mz_tol=0.02, min_cos=0.7):
    # reading mzml with pyteomics
    mzml_data = mzml.read(query_mzml_filename)
    output_results_list = []

    for spectrum in tqdm(mzml_data):
        try:
            # only selecting ms2
            ms_level = spectrum.get("ms level", 1)
            
            if ms_level != 2:
                continue

            # getting scan
            spectrum_id = spectrum.get('id', '')

            spectrum_id = _get_scan(spectrum_id)

            mz = spectrum["m/z array"]
            intensity = spectrum["intensity array"]

            # casting to floats
            mz = [float(x) for x in mz]
            intensity = [float(x) for x in intensity]

            peaks_zip = list(zip(mz, intensity))

            precursor_mz = spectrum["precursorList"]["precursor"][0]["selectedIonList"]["selectedIon"][0]["selected ion m/z"]

            results_dict = fasst.query_fasst_api_peaks(precursor_mz, peaks_zip, database,
                analog=analog, 
                precursor_mz_tol=precursor_mz_tol, 
                fragment_mz_tol=fragment_mz_tol, 
                min_cos=min_cos)

            if "status" in results_dict:
                print("Error in Results.")
                continue

            results_df = pd.DataFrame(results_dict["results"])

            # adding the scan number information
            results_df["query_scan"] = spectrum_id

            output_results_list.append(results_df)
        except KeyboardInterrupt:
            raise
        except:
            print("Error in Query")
            raise
            #pass

    if len(output_results_list) == 0:
        print("No results found")
        return pd.DataFrame()

    output_results_df = pd.concat(output_results_list)

    return output_results_df

def masst_query_usi_all(query_df, database, 
                            analog=False, 
                            precursor_mz_tol=0.02, 
                            fragment_mz_tol=0.02,
                            min_cos=0.7):
    output_results_list = []

    for query_element in tqdm(query_df.to_dict(orient="records")):
        try:

            if "usi" in query_element:
                usi = query_element["usi"]
            elif "USI"  in query_element:
                usi = query_element["USI"]

            results_dict = fasst.query_fasst_api_usi(usi, database,
                analog=analog, 
                precursor_mz_tol=precursor_mz_tol, 
                fragment_mz_tol=fragment_mz_tol, 
                min_cos=min_cos)
            results_df = pd.DataFrame(results_dict["results"])

            # TODO: Support munging of microbemasst results
            #if masst_type == "microbemasst":
                # Lets do additionally processing
            #    print("MICROBEMASST")
            # TODO: Merge with metadata automatically

            results_df["query_usi"] = usi
            if "flag" in query_element:
                results_df["flag"] = query_element["flag"]

            output_results_list.append(results_df)
        except:
            print("Error in Query")
            pass

    if len(output_results_list) == 0:
        print("No results found")
        return pd.DataFrame()

    output_results_df = pd.concat(output_results_list)

    return output_results_df

def main():
    parser = argparse.ArgumentParser(description='Fast MASST Client')
    parser.add_argument('input_file', help='file to query, can be csv mass spec file')
    parser.add_argument('output_file', help='output_file')

    parser.add_argument('--database', help='Type database to actually search', default="metabolomicspanrepo_index_nightly")
    parser.add_argument('--analog', help='Perform Yes or No', default="No")
    parser.add_argument('--precursor_tolerance', help='precursor_tolerance', default=0.02, type=float)
    parser.add_argument('--fragment_tolerance', help='fragment_tolerance', default=0.02, type=float)
    parser.add_argument('--cosine', help='cosine', default=0.7, type=float)

    # parallel throughput
    # TODO: IMPLEMENT THIS
    parser.add_argument('--querymode', help='should we do sychronous or batches', default="sync", choices=["sync", "batch"])

    args = parser.parse_args()

    analog_boolean = args.analog == "Yes"

    # checking extension
    if args.input_file.endswith(".csv") or args.input_file.endswith(".tsv"):
        try:
            query_df = pd.read_csv(args.input_file, sep=None)
        except:
            query_df = pd.DataFrame()
            print("Error reading input file")

        # checking if usi in header
        if "usi" not in query_df.columns:
            # lets try to load with explicit extensions
            if args.input_file.endswith(".csv"):
                query_df = pd.read_csv(args.input_file, sep=",")
            elif args.input_file.endswith(".tsv"):
                query_df = pd.read_csv(args.input_file, sep="\t")
            else:
                print("Unsupported file format")
                sys.exit(1)

        output_results_df = masst_query_usi_all(query_df, 
                                        args.database, 
                                        analog=analog_boolean,
                                        precursor_mz_tol=args.precursor_tolerance,
                                        fragment_mz_tol=args.fragment_tolerance)

    elif args.input_file.endswith(".mgf"):
        print("READING THE MGF FILE")
        # output_results_df = masst_query_mgf_all(args.input_file, 
        #                                 args.database, 
        #                                 analog=analog_boolean,
        #                                 precursor_mz_tol=args.precursor_tolerance,
        #                                 fragment_mz_tol=args.fragment_tolerance)
        queries_list = prep_query_mgf_queries(args.input_file,
                                                args.database, 
                                                analog=analog_boolean,
                                                precursor_mz_tol=args.precursor_tolerance,
                                                fragment_mz_tol=args.fragment_tolerance)

    elif args.input_file.lower().endswith(".mzml"):
        print("READING THE MZML FILE")
        output_results_df = masst_query_mzml_all(args.input_file,
                                        args.database,
                                        analog=analog_boolean,
                                        precursor_mz_tol=args.precursor_tolerance,
                                        fragment_mz_tol=args.fragment_tolerance)

    # lets actually execute it
    output_results_df = execute_all_queries(queries_list)

    output_results_df.to_csv(args.output_file, index=False, sep="\t")

if __name__ == '__main__':
    main()
    