from dhfcorr.ml.setup_sge_h2o import setup_h2o_cluster_sge
import argparse

if __name__ == '__main__':
    print("Training model in the cluster (SGE)")

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Name of the configuration")
    parser.add_argument("--yaml_file", default=None, help='YAML file with the configurations of the analysis.')
    parser.add_argument("--n_cores", default=None, help='Number of cores used to submit the jobs')
    args = parser.parse_args()
