import dhfcorr.config_yaml
import dhfcorr.selection as sl
import dhfcorr.selection_ml as slml

import dhfcorr.data_reader as dr

import matplotlib as mpl
import seaborn as sns
import pandas as pd

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

for run in run_list:

    file = dhfcorr.config_yaml.ConfigYaml(qa_file)
    e_cuts = sl.Cuts(file, 'electron')
    d_cuts = sl.Cuts(file, 'D0')

    ele = dr.load(config_name, "electron", run)
    # sl.build_add_features_electron(ele, e_cuts)

    # TODO: CHANGE TO MESON. Requires changes in the file structure.
    d_meson = dr.load(config_name, "d_meson", run)
    if (d_meson is None) or (ele is None):
        continue
    # sl.build_add_features_dmeson(d_meson, d_cuts)

    selected_e = sl.filter_in_pt_bins(ele, e_cuts, add_pt_bin_feat=True)
    selected_d = sl.filter_in_pt_bins(d_meson, d_cuts, add_pt_bin_feat=True)
    # selected_d = slml.select_using_ml(d_meson, 'selection_bdt.joblib', 0.124867, d_cuts)
    data_per_run.append([run, selected_e, selected_d])

data_per_run = pd.DataFrame(data_per_run, columns=['run', 'electron', 'd_meson_sel'])

merge_df = pd.concat(list(data_per_run['d_meson_sel']))
merge_df.to_hdf('filtered.hdf', 'D0')
merge_df_e = pd.concat(list(data_per_run['electron']))
merge_df_e.to_hdf('filtered_e.hdf', 'electrons')
