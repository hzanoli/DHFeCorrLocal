from __future__ import print_function

import pandas as pd
import seaborn as sns

import dhfcorr.correlate as corr
import dhfcorr.io.data_reader as reader
import dhfcorr.config_yaml as confyaml

use_built_pairs = True

sns.set()
sns.set_context('notebook')
sns.set_palette('Set1')

variables_to_keep_trig = ['GridPID', 'EventNumber', 'ID', 'IsParticleCandidate', 'Pt', 'Eta', 'Phi', 'InvMass',
                          'prediction']
variables_to_keep_assoc = ['GridPID', 'EventNumber', 'Charge', 'Pt', 'Eta', 'Phi', 'InvMassPartnersULS',
                           'InvMassPartnersLS']
index = ['GridPID', 'EventNumber']

df = reader.load('D0_HMV0', ['dmeson', 'electron'], columns=[variables_to_keep_trig, variables_to_keep_assoc],
                 index=index, lazy=True)

config_corr = confyaml.ConfigYaml()
pt_bins_trig = config_corr.values['correlation']['bins_trig']
pt_bins_assoc = config_corr.values['correlation']['bins_assoc']
trig_suffix = '_t'
assoc_suffix = '_a'

inv_mass_trig_list = list()

if use_built_pairs:
    print("Reading pairs from file")
    sum_pairs = reader.load_pairs(config_corr, 'selected')
else:
    print("Building pairs")
    sum_pairs = corr.build_pairs_from_lazy(df, (trig_suffix, assoc_suffix), pt_bins_trig, pt_bins_assoc,
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
    lambda x: corr.correlation_dmeson(x, fits, suffixes=('_t', '_a'), plot=True,
                                      subtract_non_hfe=True, **config_corr.values))
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
