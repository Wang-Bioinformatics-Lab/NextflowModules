#!/usr/bin/python
import sys
import os
import json
import argparse
from collections import defaultdict
import csv
import pandas as pd
import glob
import shutil
import requests
import yaml
import uuid

def _determine_ms_filename(download_url):
    if "metabolomicsworkbench.org" in download_url:
        # Lets parse the arguments, using urlparse
        from urllib.parse import urlparse, parse_qs
        parsed_params = urlparse(download_url)
        filename = parse_qs(parsed_params.query)['F'][0]

        return os.path.basename(filename)

    # TODO: Work for MassIVE
    # TODO: Work for GNPS
    # TODO: Work for PRIDE
    # TODO: Work for Metabolights

    return os.path.basename(download_url)

        

def main():
    parser = argparse.ArgumentParser(description='Running')
    parser.add_argument('input_download_file', help='input_download_file')
    parser.add_argument('output_file', help='output_file')
    args = parser.parse_args()

    # checking the input file exists
    if not os.path.isfile(args.input_download_file):
        print("Input file does not exist")
        exit(0)

    # Checking the file extension
    if args.input_download_file.endswith(".yaml"):
        # Loading yaml file
        parameters = yaml.load(open(args.input_download_file), Loader=yaml.SafeLoader)
        usi_list = parameters["usi"].split("\n")
    elif args.input_download_file.endswith(".tsv"):
        df = pd.read_csv(args.input_download_file, sep="\t")
        usi_list = df["usi"].tolist()

    # filtering the list to make sure non-empty usi 
    usi_list = [usi for usi in usi_list if len(usi) > 5]

    # Writing output
    with open(args.output_file, "w") as output_file:
        usi_df = pd.DataFrame()
        usi_df["usi"] = usi_list
        usi_df["flag"] = "usi_entry"

        usi_df.to_csv(output_file, index=False, sep="\t")

if __name__ == "__main__":
    main()
