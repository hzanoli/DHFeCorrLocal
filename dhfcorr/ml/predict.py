import argparse
import os
import warnings

import lightgbm as lgb
import pandas as pd
import treelite as tl
import treelite.runtime as tlrt
from tqdm import tqdm

import dhfcorr.definitions as definitions
import dhfcorr.io.data_reader as dr
from dhfcorr.config_yaml import ConfigYaml


def compile_models_treelite(files):
    path = ''.join([e + '/' for e in files[0].split('/')[:-1]])

    for file in files:
        model = tl.Model.load(file, model_format='lightgbm')
        name = path + ''.join(file.split('/')[-1].split('.')[0]) + '.so'

        try:
            os.remove(name)
        except FileNotFoundError:
            pass

        model.export_lib(toolchain=definitions.TOOLCHAIN, libpath=path + name, verbose=False)


class Model:
    def __init__(self, dataset_name, prefix='', pt_bins=None, features=None, yaml_config=None,
                 use_compiled_models=False, recompile=False):

        file_models = dr.find_models(dataset_name, prefix=prefix)
        file_models_compiled = dr.find_models(dataset_name, prefix=prefix, compiled=True)

        if len(file_models) < 1 and len(file_models_compiled) < 1:
            raise FileNotFoundError('It was not possible to find any models in this dataset')

        if use_compiled_models:
            if len(file_models) != len(file_models_compiled) or recompile:
                compile_models_treelite(file_models)
                file_models_compiled = dr.find_models(dataset_name, compiled=True)

            self.models_pt = [tlrt.Predictor(x, verbose=False) for x in file_models_compiled]

        else:
            self.models_pt = [lgb.Booster(model_file=x) for x in file_models]

        if yaml_config is not None:
            self.pt_bins = yaml_config.values['model_building']['bins_pt']
            self.features = yaml_config.values['model_building']['features']
            if pt_bins is not None or features is not None:
                warnings.warn('The Pt bins and features passed were ignored. Using the ones from the yaml file.')
        else:
            self.pt_bins = pt_bins
            self.features = features

    def predict_one_pt_bin(self, data, pt_bin):
        data_to_predict = data[self.features]

        if not isinstance(self.models_pt[0], lgb.Booster):
            data_to_predict = tlrt.Batch.from_npy2d(data_to_predict.values)

        return pd.DataFrame({'Probability': self.models_pt[pt_bin].predict(data_to_predict)},
                            index=data_to_predict.index)

    def predict(self, data):
        pt_bins = pd.cut(data.Pt, self.pt_bins, labels=False)
        return data[self.features].groupby(pt_bins, group_keys=False, as_index=False).apply(
            lambda x: self.predict_one_pt_bin(x, int(x.name)))


def predict_class(files, yaml_file, prefix):
    dataset_name = dr.get_dataset_name_from_file(files[0])
    config = ConfigYaml(yaml_file)
    gbm = Model(dataset_name, yaml_config=config, prefix=prefix)
    location_to_save = dr.get_location_step(dataset_name, 'consolidated')

    for file in tqdm(files):
        data = pd.read_parquet(file)
        data['Probability'] = gbm.predict(data)['Probability']
        file_name = location_to_save + dr.get_file_name(file)

        try:
            os.remove(file_name)
        except FileNotFoundError:
            pass

        data.to_parquet(file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("files", help="csv file with the file list")
    parser.add_argument("--prefix", default='', help='Prefix when saving the model files')
    parser.add_argument("--yaml_file", default=None, help='YAML file with the configurations of the analysis. If None, '
                                                          'uses the default configuration.')

    args = parser.parse_args()

    file_list = list(pd.read_csv(args.files)['file'])

    print('Processing the files: ')
    for f in file_list:
        print(f)

    predict_class(file_list, args.yaml_file, args.prefix)
    print('Processing done.')
