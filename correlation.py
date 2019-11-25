import pandas as pd
import numpy as np
import seaborn as sns

import dhfcorr.correlate as corr
import dhfcorr.io.data_reader as reader
import dhfcorr.config_yaml as confyaml

use_built_pairs = False

sns.set()
sns.set_context('notebook')
sns.set_palette('Set1')

variables_to_keep_trig = ['RunNumber', 'EventNumber', 'ID', 'IsParticleCandidate', 'Pt', 'Eta', 'Phi', 'InvMass', 'bkg']
variables_to_keep_assoc = ['RunNumber', 'EventNumber', 'Charge', 'Pt', 'Eta', 'Phi', 'InvMassPartnersULS',
                           'InvMassPartnersLS']
index = ['RunNumber', 'EventNumber']

electron = reader.load('D0_HMV0', 'electron', columns=variables_to_keep_assoc, index=index, lazy=True)
dmeson = reader.load('D0_HMV0', 'dmeson', columns=variables_to_keep_trig, index=index, lazy=True)
df = list(zip(dmeson, electron))

config_corr = confyaml.ConfigYaml()
pt_bins_trig = config_corr.values['correlation']['bins_trig']
pt_bins_assoc = config_corr.values['correlation']['bins_assoc']
trig_suffix = '_t'
assoc_suffix = '_a'

pt_bins_trig_mid = (np.array(pt_bins_trig) + np.array(pt_bins_trig)) / 2
pt_bins_pd_format = list(pd.cut(pt_bins_trig_mid, pt_bins_trig).categories)
best_sig_cuts = [0.022, 0.035, 0.04, 0.0895, 0.1875, 0.312, 0.3785, 0.499, 0.5, 0.497, 0.4295]
best_sig_cuts_dict = dict(zip(pt_bins_pd_format, best_sig_cuts))


def filter_dmeson(data, cuts):
    return data[data['bkg'] < cuts[data.name]]


inv_mass_trig_list = list()

if use_built_pairs:
    print("Reading pairs from file")
    sum_pairs = reader.load_pairs(config_corr, 'selected')
else:
    print("Building pairs")

    sum_pairs = corr.build_pairs_from_lazy(df, (trig_suffix, assoc_suffix), pt_bins_trig, pt_bins_assoc,
                                           filter_trig=lambda x: filter_dmeson(x, best_sig_cuts_dict),
                                           **config_corr.values['correlation'])

    sum_pairs.to_parquet('pairs_d_hfe_hm.pkl')

print("Recalculating the Phi and Pt Bins")
sum_pairs['DeltaPhiBin'] = pd.cut(sum_pairs['DeltaPhi'], config_corr.values['correlation']['bins_phi'])
sum_pairs['DeltaEtaBin'] = pd.cut(sum_pairs['DeltaEta'], config_corr.values['correlation']['bins_eta'])
sum_pairs['APtBin'] = pd.cut(sum_pairs['Pt_a'], config_corr.values['correlation']['bins_assoc'])
sum_pairs['TPtBin'] = pd.cut(sum_pairs['Pt_t'], config_corr.values['correlation']['bins_trig'])

# Get ULS and LS pair numbers
print("Calculating the NHFe pairs")
sum_pairs['NULS_a'] = sum_pairs.InvMassPartnersULS_a.transform(lambda x: len(x[x < 0.14]))
sum_pairs['NLS_a'] = sum_pairs.InvMassPartnersLS_a.transform(lambda x: len(x[x < 0.14]))

fits = pd.read_pickle(config_corr.values['base_folder'] + '/' + 'fits_inv_mass.pkl')

correlation_dist = sum_pairs.groupby(['APtBin', 'TPtBin']).apply(
    lambda x: corr.correlation_dmeson(x, suffixes=('_t', '_a'), plot=True,
                                      subtract_non_hfe=False, **config_corr.values))
correlation_dist.reset_index(inplace=True)
name = 'hmv0'
correlation_dist.to_pickle(name + '_results_correlation_inc.pkl')

"""" 
# D - ULS electron correlation
d_uls = sum_pairs.loc[sum_pairs['NULS_a']>0]
correlation_dist_uls = d_uls.groupby(['APtBin', 'TPtBin']).apply(
    lambda x: corr.correlation_dmeson(x, suffixes=('_t', '_a'), plot=True, **config_corr.correlation))
correlation_dist_uls.reset_index(inplace=True)
name = 'hmv0'
correlation_dist_uls.to_pickle(name + '_results_correlation_uls.pkl')

# D - LS electron correlation
d_ls = sum_pairs.loc[sum_pairs['NLS_a']>0]
correlation_dist_ls = d_ls.groupby(['APtBin', 'TPtBin']).apply(
    lambda x: corr.correlation_dmeson(x, suffixes=('_t', '_a'), plot=True, **config_corr.correlation))
correlation_dist_ls.reset_index(inplace=True)
name = 'hmv0'
correlation_dist_ls.to_pickle(name + '_results_correlation_ls.pkl')
"""
