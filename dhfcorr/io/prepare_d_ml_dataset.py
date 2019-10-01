import dhfcorr.io.data_reader as reader
import dhfcorr.selection.selection as sl
import dhfcorr.config_yaml as configyaml
import numpy as np
import pandas as pd
import dhfcorr.definitions as definitions
import os
import argparse
import subprocess


def calculate_mean_sigma_mc(df):
    mc_shape = df.groupby(['PtBin'])['InvMass'].agg({'mean', 'std', 'count'}).reset_index()
    mc_shape['PtMin'] = mc_shape['PtBin'].apply(lambda x: x.left).astype(np.float)
    mc_shape['PtMax'] = mc_shape['PtBin'].apply(lambda x: x.right).astype(np.float)
    mc_shape['PtMid'] = mc_shape['PtBin'].apply(lambda x: x.mid).astype(np.float)
    mc_shape['XErr'] = mc_shape['PtBin'].apply(lambda x: x.length / 2.).astype(np.float)
    mc_shape.set_index('PtBin', inplace=True)
    return mc_shape


if __name__ == '__main__':
    print("Preparing the data sample for training. This step will generate a dataset with signal candidates and "
          "submit jobs for the background.")

    parser = argparse.ArgumentParser()
    parser.add_argument("mc_config", help="Name of the configuration used in MC (used for signal).")
    parser.add_argument("data_config", help="Name of the configuration used in data (used for background).")
    # parser.add_argument("--meson", choices=['D0', 'D+', 'Dstar'], default='D0', help='D meson that will be used.')
    parser.add_argument("--yaml_file", default=None, help='YAML file with the configurations of the analysis.')
    parser.add_argument("--nfiles", help='Number of files per job.', default=70)

    args = parser.parse_args()

    print("The following configuration will be used:")
    print('Configuration in MC (for signal): ' + args.mc_config)
    print('Configuration in data (for background): ' + args.data_config)

    d_cuts = configyaml.ConfigYaml(args.yaml_file)

    cols_to_load = d_cuts.values['model_building']['features'] + d_cuts.values['model_building']['additional_features']
    cols_to_load_mc = cols_to_load + d_cuts.values['model_building']['additional_features_mc']
    signal = sl.get_true_dmesons(reader.load(args.mc_config, 'dmeson'))[cols_to_load_mc]
    signal['PtBin'] = pd.cut(signal['Pt'], d_cuts.values['model_building']['bins_pt'])

    # Create the columns which will be used for the ML training 'CandidateType'
    # CandidateType = -1 -> Background
    # CandidateType =  0 -> Non-Prompt D mesons
    # CandidateType =  1 -> Prompt D mesons
    signal['CandidateType'] = signal['IsPrompt'].astype(np.int)

    processing_folder = definitions.PROCESSING_FOLDER + args.data_config
    folder_to_save = processing_folder + '/ml-dataset/'

    signal.groupby('PtBin', as_index=False).apply(lambda x: x.drop('PtBin', axis='columns').to_parquet(
        folder_to_save + 'sig_' + str(x.name) + '.parquet'))
    mc_mean_sigma = calculate_mean_sigma_mc(signal)

    if not os.path.isdir(processing_folder):
        os.mkdir(processing_folder)
    if not os.path.isdir(processing_folder + '/ml-dataset'):
        os.mkdir(processing_folder + '/ml-dataset/')

    mc_mean_sigma.to_pickle(processing_folder + '/ml-dataset/mc_mean_sigma.pkl')

    from dhfcorr.io.utils import batch, format_list_to_bash

    runs = reader.get_run_list(args.data_config)

    job_id = 0
    for run_list in batch(runs, args.nfiles):
        job_name = 'bkg4ml_' + str(job_id)

        submit_part = r"qsub -V -cwd -N " + job_name + " -S $(which python3) " + definitions.ROOT_DIR + \
                      '/io/create_bkg_sample.py'
        command = submit_part + ' ' + format_list_to_bash(run_list) + ' ' + args.data_config + ' --id ' + str(job_id)
        if args.yaml_file is not None:
            command += ' --yaml_file ' + args.yaml_file
        print()
        print("Submitting job " + str(job_id))
        subprocess.run(command, shell=True)
        job_id = job_id + 1
