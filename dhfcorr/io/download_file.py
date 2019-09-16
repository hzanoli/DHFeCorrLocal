#!/usr/bin/env python

import subprocess


def download_file(grid_path, local_path):
    grid_path = grid_path.split(',')
    local_path = local_path.split(',')

    for grid, local in zip(grid_path, local_path):
        command = 'alien_cp  alien:' + str(grid) + ' file:' + local
        subprocess.run(command.split())


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("grid_path", help='List with the location of the files on alien(grid)')
    parser.add_argument("local_path", help='List with the location (and names) that the files will be saved')

    args = parser.parse_args()
    download_file(args.grid_path, args.local_path)
