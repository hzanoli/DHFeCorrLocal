import pandas as pd
import argparse

if __name__ == '__main__':
    print("Utility to merge the MC dataset into a single file.")

    parser = argparse.ArgumentParser()
    parser.add_argument("mc_config", help="Name of the dataset used in MC (used for signal).")

    args = parser.parse_args()

    signal = reader.load(args.mc_config, 'dmeson')
