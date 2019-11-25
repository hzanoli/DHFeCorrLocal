#!/usr/bin/env python
import pandas as pd
import subprocess
import os


def download_file(grid_path, local_path):
    for grid, local in zip(grid_path, local_path):
        command = 'alien_cp  alien:' + str(grid) + ' file:' + local
        subprocess.run(command.split())


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("login", help='CSV file used to login')
    parser.add_argument("csv_file", help='CSV file with locations from grid and local')

    args = parser.parse_args()
    login = pd.read_csv(args.login).iloc[0]
    user = login['user']
    code = login['code']
    token = subprocess.Popen('echo ' + code + ' | alien-token-init ' + user, shell=True,
                             stdout=subprocess.PIPE)
    token.wait()
    files = pd.read_csv(args.csv_file)
    download_file(files['grid'], files['local'])

    os.remove(args.csv_file)
