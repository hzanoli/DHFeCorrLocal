import glob
import argparse


def is_processed(f):


if __name__ == 'main':
    parser = argparse.ArgumentParser()
    parser.add_argument("job_name", help="Job name to be validated")
    args = parser.parse_args()
    job_files = glob.glob(args.job_name + '*')

    for file in job_files:
        with open(file, "r") as f:
            for line in f:
                for "Processing done." in line.split()
