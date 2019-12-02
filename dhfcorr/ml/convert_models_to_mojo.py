import h2o
import argparse
import dhfcorr.definitions as definitions
import glob
import subprocess

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Configuration name")
    parser.add_argument("--prefix", default='', help='Prefix when saving the model files')
    args = parser.parse_args()

    dataset = args.dataset
    prefix = args.prefix

    h2o.init()
    print('Converting the models to MOJOs')
    path_model_files = definitions.PROCESSING_FOLDER + dataset + '/ml-dataset/'
    model_files = glob.glob(path_model_files + prefix + 'model_pt*_main')

    print('The following models were found:')
    print(model_files)

    print()

    for file in model_files:
        model = h2o.load_model(file)
        model_name = path_model_files + file.split('/')[-1] + '_mojo.zip'
        path_saved = model.download_mojo('~/temp_mojos/', get_genmodel_jar=True)
        subprocess.run('mv ' + path_saved + ' ' + model_name, shell=True)

        print('Saving ' + file + ' to ' + model_name)

    print('Processing Done.')
