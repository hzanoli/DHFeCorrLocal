import ROOT
import numpy as np
import pandas as pd
import seaborn as sns

import dhfcorr.config_yaml as configyaml
import dhfcorr.correlate as corr
import dhfcorr.io.data_reader as reader

variables_to_keep_trig = ['GridPID', 'EventNumber', 'ID', 'IsParticleCandidate', 'Pt', 'Eta', 'Phi', 'InvMass',
                          'prediction']
variables_to_keep_assoc = ['GridPID', 'EventNumber', 'Charge', 'Pt', 'Eta', 'Phi', 'InvMassPartnersULS',
                           'InvMassPartnersLS']
index = ['GridPID', 'EventNumber']

df = reader.load('D0_HMV0', ['dmeson', 'electron'], columns=variables_to_keep_trig, index=index, lazy=True)

config_corr = configyaml.ConfigYaml('dhfcorr/config/optimize_bdt_cut.yaml')

pt_bins_trig = config_corr.values['correlation']['bins_trig']
pt_bins_assoc = config_corr.values['correlation']['bins_assoc']
trig_suffix = '_t'
assoc_suffix = '_a'

inv_mass_trig_list = list()

pairs = corr.build_pairs_from_lazy(df, (trig_suffix, assoc_suffix), pt_bins_trig, pt_bins_assoc,
                                   **config_corr.values['correlation'])

selected = pd.read_pickle('pairs_d_hfe_hm.pkl').reset_index(level=0, drop=True)

# Remove ROOT messages
ROOT.gPrintViaErrorHandler = ROOT.kTRUE
ROOT.gErrorIgnoreLevel = ROOT.kWarning

values_to_cut_dbt = np.linspace(0.8, 1.0, 200, endpoint=False)

result_optimization = list()
for cut in values_to_cut_dbt:
    print()
    print("Changing the bdt output cut to:")
    print(cut)
    print()
    selected = selected[selected['prediction' + trig_suffix] > cut]
    correlation_dist = selected.groupby(['APtBin', 'TPtBin']).apply(
        lambda x: corr.correlation_dmeson(x, suffixes=('_t', '_a'), mix=False, **config_corr.values['correlation']))


    def get_error(x, percent=False):
        try:
            if percent:
                return x.data['Error'].iloc[0] / x.data['Content'].iloc[0]
            else:
                return x.data['Error'].iloc[0]
        except IndexError:
            return np.nan


    correlation_dist['Error_corr'] = correlation_dist['DMesonCorr'].apply(get_error)
    correlation_dist['Error_corr_percent'] = correlation_dist['DMesonCorr'].apply(get_error, percent=True)
    correlation_dist['Cut'] = cut

    result_optimization.append(correlation_dist)

result = pd.concat(result_optimization)

result.to_pickle('optimization_bdt_cut.pkl')
