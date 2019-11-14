import argparse
import dhfcorr.config_yaml as configyaml
import dhfcorr.definitions as definitions
import numpy as np
import pandas as pd
import glob
import os
from tqdm import tqdm


def merge(base_folder_name, pt_bin, particle):
    print('Processing pt bin ' + str(pt_bin))
    files = glob.glob(base_folder_name + 'id*pt' + str(pt_bin) + '_' + particle + '.parquet')
    merged = pd.concat([pd.read_parquet(f) for f in tqdm(files, unit='files')])
    merged.to_parquet(base_folder_name + 'merged_pt' + str(pt_bin) + '_' + particle + '.parquet')

    print('Removing temporary files for pt bin ' + str(pt_bin))
    for f in tqdm(files, unit='files'):
        os.remove(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Configuration name")
    parser.add_argument("--particle", default='dmeson', help="Name of the particle")
    parser.add_argument("--yaml_file", default=None, help='YAML file with the configurations of the analysis. If None, '
                                                          'uses the default configuration.')
    args = parser.parse_args()

    d_cuts = configyaml.ConfigYaml(args.yaml_file)
    base_name = definitions.PROCESSING_FOLDER + args.config + '/skimmed/'
    pt_bins = np.array(d_cuts.values['reduce_data']['bins_pt'])

    for pt_bin_number in range(len(pt_bins) - 1):
        merge(base_name, pt_bin_number, args.particle)
