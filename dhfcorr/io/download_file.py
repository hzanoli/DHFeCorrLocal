#!/usr/bin/env python
import subprocess

import pandas as pd

from dhfcorr.cluster import get_token


def download_file_list(grid_path, local_path):
    for grid, local in zip(grid_path, local_path):
        download_file(grid, local, False)


def download_file(grid_path, local_path, silent=True):
    command = 'alien_cp  alien:' + str(grid_path) + ' file:' + local_path
    if silent:
        download = subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        download = subprocess.Popen(command, shell=True)
    download.wait()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("login", help='CSV file used to login')
    parser.add_argument("csv_file", help='CSV file with locations from grid and local')

    args = parser.parse_args()
    login = pd.read_csv(args.login).iloc[0]

    user = login['user']
    code = login['code']

    get_token(code, user)

    files = pd.read_csv(args.csv_file)
    download_file_list(files['grid'], files['local'])