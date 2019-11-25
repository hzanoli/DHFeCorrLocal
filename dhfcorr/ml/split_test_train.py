import numpy as np
import pandas as pd
import argparse
import dhfcorr.config_yaml as configyaml
import dhfcorr.definitions as definitions
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("config", help='YAML file with the configurations of the analysis.')
parser.add_argument("--yaml_file", default=None, help='YAML file with the configurations of the analysis.')
parser.add_argument("--max_examples", default=400000, help='Maximum number of examples for signal')

args = parser.parse_args()

d_cuts = configyaml.ConfigYaml(args.yaml_file)
folder_saved = definitions.PROCESSING_FOLDER + args.config + '/ml-dataset/'

pt_bins = np.array(d_cuts.values['model_building']['bins_pt'])
pt_bins = pd.cut(0.5 * (pt_bins[:-1] + pt_bins[1:]), bins=pt_bins)


def get_train_test(df, max_num):
    n_sample = int(min([max_num, 0.5 * len(df)]))
    train_ = df.iloc[:n_sample]
    test_ = df.iloc[n_sample:]
    return train_, test_


for pt_bin in range(len(pt_bins)):
    print("Processing pt bin " + str(pt_bin))
    dataset = pd.read_parquet(
        definitions.PROCESSING_FOLDER + args.config + '/ml-dataset/ml_sample_' + str(pt_bin) + '.parquet')

    dataset = dataset.sample(frac=1., random_state=1313).reset_index(drop=True)

    # for training, limit the total amount of signal to max_examples or 0.5, whatever is smaller
    prompt_train, prompt_test = get_train_test(dataset[dataset['CandidateType'] == 1], args.max_examples)
    n_prompt_train, n_prompt_test = get_train_test(dataset[dataset['CandidateType'] == 0], args.max_examples)
    bkg_train, bkg_test = get_train_test(dataset[dataset['CandidateType'] == -1], 2 * args.max_examples)

    # No need to separate now the y: it is all kept in the parquet file
    train = pd.concat([prompt_train, n_prompt_train, bkg_train],
                      ignore_index=True).sample(frac=1., random_state=1313).reset_index(drop=True)
    test = pd.concat([prompt_test, n_prompt_test, bkg_test], ignore_index=True).sample(frac=1,
                                                                                       random_state=1313).reset_index(
        drop=True)

    train.to_parquet(
        definitions.PROCESSING_FOLDER + args.config + '/ml-dataset/ml_sample_train_' + str(pt_bin) + '.parquet')
    test.to_parquet(
        definitions.PROCESSING_FOLDER + args.config + '/ml-dataset/ml_sample_test_' + str(pt_bin) + '.parquet')
