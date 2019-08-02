import dhfcorr.config_yaml
import dhfcorr.selection as sl
import dhfcorr.selection_ml as slml

import dhfcorr.data_reader as dr

import matplotlib as mpl
import seaborn as sns
import pandas as pd
import numpy as np
from joblib import load

mpl.rc('image', cmap='Reds')
sns.set()
sns.set_context("talk")
sns.set_palette('Reds')
mpl.rc('image', cmap='Reds')

run_list = [265525, 265521, 265501, 265500, 265499, 265435, 265427, 265426, 265425,
            265424, 265422, 265421, 265420, 265419, 265388, 265387, 265385, 265384, 265383, 265381, 265378,
            265377, 265344, 265343, 265342, 265339, 265338, 265336, 265334, 265332, 265309]

data_per_run = list()
qa_file = 'default_config_local.yaml'
config_name = 'tree_v2'

bins_ml = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10., 12., 16., np.inf]

for run in run_list:
    try:
        file = dhfcorr.config_yaml.ConfigYaml(qa_file)
        e_cuts = sl.Cuts(file, 'electron')
        d_cuts = sl.Cuts(file, 'D0')

        ele = dr.load(config_name, "electron", run)
        sl.build_add_features_electron(ele, e_cuts)

        d_meson = dr.load(config_name, "d_meson", run)

        if d_meson is None:
            continue

        sl.build_add_features_dmeson(d_meson, d_cuts)

        selected_e = sl.filter_in_pt_bins(ele, e_cuts, add_pt_bin_feat=True)
        d_meson['PtBinML'] = pd.cut(d_meson['Pt'], bins=bins_ml, labels=False)
        selected_d = d_meson.groupby(by='PtBinML').apply(lambda x: slml.decision_function(x))
        # selected_d = d_meson

        data_per_run.append([run, selected_e, selected_d])
    except FileNotFoundError as err:
        print(err)
        print('File not found for run {0}'.format(str(run)))

data_per_run = pd.DataFrame(data_per_run, columns=['run', 'electron', 'd_meson_sel'])

merge_df = pd.concat(list(data_per_run['d_meson_sel']))
merge_df.to_hdf('filtered.hdf', 'D0')
merge_df_e = pd.concat(list(data_per_run['electron']))
merge_df_e.to_hdf('filtered_e.hdf', 'electrons')
