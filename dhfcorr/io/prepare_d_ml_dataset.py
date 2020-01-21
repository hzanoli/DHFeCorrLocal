import argparse
import subprocess

from tqdm import tqdm

import dhfcorr.config_yaml as configyaml
import dhfcorr.definitions as definitions
import dhfcorr.io.data_reader as dr
from dhfcorr.cluster import get_job_command
from dhfcorr.io.prepare_signal import prepare_signal

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
    parser.add_argument('-s', "--skip_signal", dest='skip_signal', action='store_true', help='Skip signal processing')
    parser.set_defaults(submit_bkg=True)
    parser.set_defaults(skip_signal=False)

    args = parser.parse_args()

    print("The following configuration will be used:")
    print('Configuration in MC (for signal): ' + args.mc_config)
    print('Configuration in data (for background): ' + args.data_config)

    d_cuts = configyaml.ConfigYaml(args.yaml_file)

    dr.check_for_folder(dr.get_location_step(args.data_config, 'ml'))

    if not args.skip_signal:
        prepare_signal(args.mc_config, d_cuts.values['model_building']['bins_pt'], 'dmeson')

    from dhfcorr.utils import batch, format_list_to_bash

    runs = dr.get_run_numbers(args.data_config)

    print("Processing Background:")
    clear = subprocess.Popen('rm -f ' + ' bkg_*', shell=True)
    clear.wait()
    job_id = 0

    for run_list in tqdm(list(batch(runs, args.nfiles))):
        job_name = args.data_config + '_bkg_' + str(job_id)

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
