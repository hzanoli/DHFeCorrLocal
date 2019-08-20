import ROOT
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import root_numpy as rnp
import dhfcorr.correlate as corr
from histogram.histogram import Histogram
import dhfcorr.io.data_reader as reader

sns.set()
sns.set_context('notebook')
sns.set_palette('Set1')
ROOT.TH1.AddDirectory(False)

variables_to_keep_D = ['GridPID', 'EventNumber', 'ID', 'IsParticleCandidate', 'Pt', 'Eta', 'Phi', 'InvMass']

variables_to_keep_e = ['GridPID', 'EventNumber', 'Centrality', 'VtxZ', 'Charge', 'Pt', 'Eta', 'Phi',
                       'InvMassPartnersULS', 'InvMassPartnersLS']

selection_type = 'ML'
df_e = pd.read_parquet('selected_e.parquet')
df_e = df_e[variables_to_keep_e]

name = 'cut_bdt'  #

if selection_type == 'ML':
    df_d = pd.read_parquet('selected_ml.parquet')
    cuts_pt = pd.read_pickle(name + '.pkl')
    df_d = df_d[df_d['Pt'] < 24.]
    df_d['Cut'] = cuts_pt.loc[df_d['Pt']].reset_index(drop=True)
    df_d = df_d[df_d['prediction'] >= df_d['Cut']]
    variables_to_keep_D = variables_to_keep_D + ['prediction']
else:
    df_d = pd.read_parquet('selected_rectangular.parquet')

if 'InvMassD0' in df_d.columns:
    df_d['InvMass'] = df_d['InvMassD0']
df_d = df_d[variables_to_keep_D]

reader.reduce_dataframe_memory(df_d)
reader.reduce_dataframe_memory(df_e)

config_corr = corr.CorrConfig('dhfcorr/config/default_config_local.yaml')
# First create the pairs with all the selected D mesons and Electrons
# This is necessary because since both D meson and electron where selected.
# So we need to check if they are still in the same event.

pairs = corr.build_pairs(df_d, df_e, **config_corr.correlation)
del df_d, df_e

print(pairs.info())

correlation_dist = pairs.groupby(['EPtBin', 'DPtBin']).apply(
    lambda x: corr.correlation_dmeson(x, config_corr=config_corr, plot=True))
correlation_dist.reset_index(inplace=True)

# print(correlation_dist)

ROOT.gPrintViaErrorHandler = ROOT.kTRUE
ROOT.gErrorIgnoreLevel = ROOT.kWarning

# result_optimization = list()
# for cut in values_to_cut_dbt:
#    print(cut)
#    selected = pairs[pairs['prediction_d'] > cut]
#    correlation_dist = selected.groupby(['EPtBin', 'DPtBin']).apply(
#        lambda x: corr.correlation_dmeson(x, config_corr=config_corr))
#    correlation_dist['Error_corr'] = correlation_dist['DMesonCorr'].apply(lambda x: x.data['Error'].iloc[0])
#    correlation_dist['Error_corr_percent'] = correlation_dist['DMesonCorr'].apply(lambda x: x.data['Error'].iloc[0]/x.data['Content'].iloc[0])
#    correlation_dist['Cut'] = cut
#    result_optimization.append(correlation_dist)

# result = pd.concat(result_optimization)
# result.to_pickle('optimization_bdt_cut.pkl')

if selection_type == 'ML':
    correlation_dist.to_pickle(name + '_results_correlation.pkl')
else:
    correlation_dist.to_pickle(name + '_results_correlation_rec.pkl')


# for mass_fit, e_i, d_i in zip(correlation_dist['InvMassFit'], correlation_dist['EPtBin'], correlation_dist['DPtBin']):
#    mass_fit.get_figure().savefig('mass_pt_e' + str(e_i) + '_d' + str(d_i) + '.pdf', bbox_inches="tight")

# for corr_d, e_i, d_i in zip(correlation_dist['DMesonCorr'], correlation_dist['EPtBin'], correlation_dist['DPtBin']):
#    ax_sig = corr_d.plot1d('DeltaPhiBin', label='D-Inclusive electron')
#    # ax_sig.set_ylim(min([ax_sig.get_ylim()[0], 0]), 1.2 * ax_sig.get_ylim()[1])
#    ax_sig.legend()
#    ax_sig.set_xlabel(r'$\Delta\varphi$ [rad]')
#    ax_sig.set_ylabel(r'$\frac{1}{N^{D^0}} \frac{dN}{d\Delta\varphi}$')
#    ax_sig.get_figure().savefig('correlation_pte' + str(e_i) + '_d' + str(d_i) + '.pdf', bbox_inches="tight")

def correlate_same(df_d, df_e, suffixes=('_d', '_e'), axis=None, **kwargs):
    """"Calculates the angular correlations between d mesons and electrons in the same event.

    Parameters
    ----------
    df_d : pd.DataFrame
        Dataframe with the D mesons.
    df_e : pd.DataFrame
         Dataframe with electrons.
    suffixes:
        suffix used to identify d mesons and electrons
    axis : list
        The list of axis that the same event histogram will be returned
    kwargs :
        Additional parameters used to configure the correlation.

    Returns
    -------
    histogram : hist.Histogram
        The same-event histogram
    Raises
    ------

    """

    pairs, selected_d, selected_e = corr.build_pairs(df_d, df_e, suffixes, **kwargs)

    # Fit the Invariant mass for each pt_bin (in D and E)

    # selected_mass, weights

    prefixes = [suffix[1:].upper() for suffix in suffixes]

    d_bins = selected_d['PtBin' + prefixes[0]].cat.categories
    e_bins = selected_d['PtBin' + prefixes[1]].cat.categories
