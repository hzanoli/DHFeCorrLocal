import ROOT
import numpy as np
import pandas as pd
import dhfcorr.correlate as corr

variables_to_keep_D = ['GridPID', 'EventNumber', 'ID', 'IsParticleCandidate', 'Pt', 'Eta', 'Phi', 'InvMass',
                       'prediction']

variables_to_keep_e = ['GridPID', 'EventNumber', 'Centrality', 'VtxZ', 'Charge', 'Pt', 'Eta', 'Phi',
                       'InvMassPartnersULS', 'InvMassPartnersLS']

df_e = pd.read_parquet('selected_e.parquet')
df_e = df_e[variables_to_keep_e]

df_d = pd.read_parquet('selected_ml.parquet')
cuts_pt = pd.read_pickle('cut_bdt.pkl')

df_d['InvMass'] = df_d['InvMassD0']
df_d = df_d[variables_to_keep_D]

config_corr = corr.CorrConfig('dhfcorr/config/optimize_bdt_cut.yaml')
pairs = corr.build_pairs(df_d, df_e, **config_corr.correlation)

# Calculate the invariant mass for each (D meson, electron) pT bin
values_to_cut_dbt = np.linspace(0.4, 1.0, 180, endpoint=False)

# Remove ROOT messages
ROOT.gPrintViaErrorHandler = ROOT.kTRUE
ROOT.gErrorIgnoreLevel = ROOT.kWarning

result_optimization = list()
for cut in values_to_cut_dbt:
    print(cut)
    selected = pairs[pairs['prediction_d'] > cut]
    correlation_dist = selected.groupby(['EPtBin', 'DPtBin']).apply(
        lambda x: corr.correlation_dmeson(x, config_corr=config_corr))
    correlation_dist['Error_corr'] = correlation_dist['DMesonCorr'].apply(lambda x: x.data['Error'].iloc[0])
    correlation_dist['Error_corr_percent'] = correlation_dist['DMesonCorr'].apply(
        lambda x: x.data['Error'].iloc[0] / x.data['Content'].iloc[0])
    correlation_dist['Cut'] = cut
    result_optimization.append(correlation_dist)

result = pd.concat(result_optimization)
result.to_pickle('optimization_bdt_cut.pkl')
