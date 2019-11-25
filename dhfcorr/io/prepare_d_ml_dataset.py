import dhfcorr.io.data_reader as reader
import dhfcorr.selection.selection as sl
import dhfcorr.config_yaml as configyaml
import numpy as np
import pandas as pd
import dhfcorr.definitions as definitions
from tqdm import tqdm
import os
import argparse
import subprocess
from dhfcorr.submit_job import get_job_command


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
    parser.add_argument("mc_config", help="Name of the dataset used in MC (used for signal).")
    parser.add_argument("data_config", help="Name of the dataset used in data (used for background).")
    # parser.add_argument("--meson", choices=['D0', 'D+', 'Dstar'], default='D0', help='D meson that will be used.')
    parser.add_argument("--yaml_file", default=None, help='YAML file with the configurations of the analysis. If None, '
                                                          'the default is loaded.')
    parser.add_argument("--nfiles", type=int, help='Number of files per job.', default=30)
    parser.add_argument("--nsubprocess", type=int, help='Number of subprocess for local job.', default=10)
    parser.add_argument("--submit-bkg", dest='submit_bkg', action='store_true', help='Submit the background '
                                                                                     'generation on cluster')
    parser.add_argument("--not-submit-bkg", dest='submit_bkg', action='store_false', help='DO NOT Submit the background'
                                                                                          ' generation on cluster')
    parser.add_argument('-s', "--skip_signal", dest='skip_signal', action='store_true', help='Skip sinal processing')
    parser.set_defaults(submit_bkg=True)
    parser.set_defaults(skip_signal=False)

    args = parser.parse_args()

    print("The following configuration will be used:")
    print('Configuration in MC (for signal): ' + args.mc_config)
    print('Configuration in data (for background): ' + args.data_config)

    d_cuts = configyaml.ConfigYaml(args.yaml_file)

    if not args.skip_signal:
        print("Processing signal")
        signal = sl.get_true_dmesons(reader.load(args.mc_config, 'dmeson'))
        signal['PtBin'] = pd.cut(signal['Pt'], d_cuts.values['model_building']['bins_pt'])

        # Create the columns which will be used for the ML training 'CandidateType'
        # CandidateType = -1 -> Background
        # CandidateType =  0 -> Non-Prompt D mesons
        # CandidateType =  1 -> Prompt D mesons
        signal['CandidateType'] = signal['IsPrompt'].astype(np.int)

        processing_folder = definitions.PROCESSING_FOLDER + args.data_config
        folder_to_save = processing_folder + '/ml-dataset/'

        if not os.path.isdir(processing_folder):
            os.mkdir(processing_folder)
        if not os.path.isdir(folder_to_save):
            os.mkdir(folder_to_save)

        clear = subprocess.Popen('rm -f ' + ' sig_*', shell=True)
        clear.wait()

        for name, group in signal.groupby('PtBin', as_index=False):
            print(name)
            df = group.drop(['PtBin'], axis='columns')
            df.to_parquet(folder_to_save + 'sig_' + str(name) + '.parquet')

        mc_mean_sigma = calculate_mean_sigma_mc(signal)

        clear = subprocess.Popen('rm -f mc_mean_sigma*', shell=True)
        clear.wait()

        mc_mean_sigma.to_pickle(folder_to_save + 'mc_mean_sigma.pkl')

    from dhfcorr.io.utils import batch, format_list_to_bash

    runs = reader.get_run_list(args.data_config)

    print("Processing Background:")
    clear = subprocess.Popen('rm -f ' + ' bkg_*', shell=True)
    clear.wait()
    job_id = 0
    for run_list in tqdm(list(batch(runs, args.nfiles))):
        job_name = 'bkg4ml_' + str(job_id)

        script = definitions.ROOT_DIR + '/io/create_bkg_sample.py'
        arguments = format_list_to_bash(run_list) + ' ' + args.data_config + ' --id ' + str(job_id)

        if args.yaml_file is not None:
            arguments += ' --yaml_file ' + args.yaml_file

        if args.submit_bkg:
            command = get_job_command(job_name, script, arguments)
            subprocess.run(command, shell=True)

        else:
            n_short = args.nsubprocess
            processes = list()
            sub_job_id = 0
            for short_run in batch(run_list, n_short):
                command = "python " + definitions.ROOT_DIR + '/io/create_bkg_sample.py '
                command = command + format_list_to_bash(short_run) + ' ' + args.data_config + ' --id ' + str(
                    job_id) + '_' + str(sub_job_id)
                if args.yaml_file is not None:
                    command += ' --yaml_file ' + args.yaml_file
                processes.append(subprocess.Popen(command, shell=True, stdout=subprocess.PIPE))

                sub_job_id += 1
            exit_codes = [p.wait() for p in processes]

        job_id += 1
