import ROOT
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import root_numpy as rnp
import dhfcorr.correlate as corr
from histogram.histogram import Histogram

sns.set()
sns.set_context('notebook')
ROOT.TH1.AddDirectory(False)

variables_to_keep_D = ['GridPID', 'EventNumber', 'ID', 'IsParticleCandidate', 'Pt', 'Eta', 'Phi', 'InvMass']

variables_to_keep_e = ['GridPID', 'EventNumber', 'Centrality', 'VtxZ', 'Charge', 'Pt', 'Eta', 'Phi',
                       'InvMassPartnersULS', 'InvMassPartnersLS']

df_e = pd.read_parquet('selected_e.parquet')
df_e = df_e[variables_to_keep_e]
# df_d = pd.read_parquet('selected_rectangular.parquet')
df_d = pd.read_parquet('selected_ml.parquet')
df_d = df_d[df_d['prediction'] > 0.9]
df_d['InvMass'] = df_d['InvMassD0']
df_d = df_d[variables_to_keep_D]

config_corr = corr.CorrConfig('dhfcorr/config/default_config_local.yaml')


def fit_d_meson_mass(df, n_bins='auto', min_hist=1.6, max_hist=2.2, **kwargs):
    inv_mass = df['InvMass']
    weight = df['Weight']

    if isinstance(n_bins, str):
        bins = np.array(np.histogram_bin_edges(inv_mass, bins=n_bins, range=(min_hist, max_hist)), dtype='float64')
        histogram = ROOT.TH1F("MassFit", "MassFit", len(bins) - 1, bins)
    else:
        histogram = ROOT.TH1F("MassFit", "MassFit", n_bins, min_hist, max_hist)

    histogram.Sumw2()
    rnp.fill_hist(histogram, inv_mass, weights=weight)

    fit = corr.fit_inv_mass_root(histogram, kwargs['inv_mass_lim']['default'])
    return fit


def correlation_signal_region(df, fit, n_sigma, suffix='_d', axis=()):
    mean, sigma = (fit.GetMean(), fit.GetSigma())
    signal = corr.select_inv_mass(df, mean - n_sigma * sigma, mean + n_sigma * sigma, suffix=suffix)
    signal_corr = Histogram.from_dataframe(signal, axis)

    return signal_corr


def correlation_background_region(df, fit, n_sigma_min, n_sigma_max, suffix='_d', axis=()):
    mean, sigma = (fit.GetMean(), fit.GetSigma())

    right = corr.select_inv_mass(df, mean - n_sigma_max * sigma, mean - n_sigma_min * sigma, suffix=suffix)
    left = corr.select_inv_mass(df, mean - n_sigma_max * sigma, mean - n_sigma_min * sigma, suffix=suffix)

    bkg = pd.concat([right, left])
    bkg_corr = Histogram.from_dataframe(bkg, axis)
    return bkg_corr


def get_n_bkg_sidebands(fit, n_sigma_min, n_sigma_max):
    mean, sigma = (fit.GetMean(), fit.GetSigma())

    background_sidebands_1 = ROOT.Double()
    error_bkg_sidebands_1 = ROOT.Double()
    fit.Background(mean - n_sigma_max * sigma, mean - n_sigma_min * sigma, background_sidebands_1,
                   error_bkg_sidebands_1)

    background_sidebands_2 = ROOT.Double()
    error_bkg_sidebands_2 = ROOT.Double()
    fit.Background(mean + n_sigma_min * sigma, mean + n_sigma_max * sigma, background_sidebands_2,
                   error_bkg_sidebands_2)
    background_sidebands = background_sidebands_1 + background_sidebands_2
    error_bkg_sidebands = np.sqrt(error_bkg_sidebands_1 ** 2 + error_bkg_sidebands_2 ** 2)

    return float(background_sidebands), float(error_bkg_sidebands)


def get_n_signal(fit, n_sigma):
    signal = ROOT.Double()
    err_signal = ROOT.Double()
    fit.Signal(n_sigma, signal, err_signal)
    return float(signal), float(err_signal)


def get_n_bkg(fit, n_sigma):
    bkg = ROOT.Double()
    err_bkg = ROOT.Double()
    fit.Background(n_sigma, bkg, err_bkg)
    return float(bkg), float(err_bkg)


# First create the pairs with all the selected D mesons and Electrons
# This is necessary because since both D meson and electron where selected.
# So we need to check if they are still in the same event.

pairs = corr.build_pairs(df_d, df_e, **config_corr.correlation)

# Calculate the invariant mass for each (D meson, electron) pT bin
d_pt_bins = pairs['DPtBin'].cat.categories
e_pt_bins = pairs['EPtBin'].cat.categories

# Fit mass bins
mass_fits_hist = list()

# Fit invariant mass
fig_list = list()
df_list = list()
correlation_signal = list()
correlation_bkg = list()

axis = ('CentralityBin', 'VtxZBin', 'DPtBin', 'EPtBin', 'DeltaEtaBin', 'DeltaPhiBin')


def correlation_dmeson(df_pairs, config_corr, n_sigma_sig=2., n_sigma_bkg_min=4., n_sigma_bkg_max=10.,
                       suffix='_d', axis=('CentralityBin', 'VtxZBin', 'DPtBin', 'EPtBin', 'DeltaEtaBin',
                                          'DeltaPhiBin'), d_pt_bin=None, e_pt_bin=None):
    d_in_this_pt = corr.reduce_to_single_particle(df_pairs, suffix)
    fit = fit_d_meson_mass(d_in_this_pt, **config_corr.correlation)

    signal_corr = correlation_signal_region(df_pairs, fit, n_sigma_sig, axis=axis).project(['DeltaPhiBin'])
    n_signal, err_n_signal = get_n_signal(fit, n_sigma_sig)

    bkg_corr = correlation_background_region(df_pairs, fit, n_sigma_bkg_min, n_sigma_bkg_max,
                                             axis=axis).project(['DeltaPhiBin'])

    n_bkg_sidebands, err_n_bkg_sidebands = get_n_bkg_sidebands(fit, n_sigma_bkg_min, n_sigma_bkg_max)
    n_bkg_signal_region, err_n_bkg_signal_region = get_n_bkg(fit, n_sigma_sig)

    # normalize the background correlation
    bkg_corr = bkg_corr / n_bkg_sidebands * n_bkg_signal_region

    if n_signal > 0:
        d_meson_corr = (signal_corr - bkg_corr) / n_signal
    else:
        d_meson_corr = Histogram()

    fig, ax = plt.subplots()
    corr.plot_inv_mass_fit(fit, ax, **config_corr.style_qa)

    # Problem in the automatic python bindings: the destructor is twice. Call it manually.
    result = pd.DataFrame({'InvMassFit': ax, 'DMesonCorr': d_meson_corr, 'SignalRegCorr': signal_corr,
                           'NSignal': n_signal, 'NSignalErr': err_n_signal, 'BkgRegCorr': bkg_corr,
                           'NBkgSideBands': n_bkg_sidebands, 'NBkgSideBandsErr': err_n_bkg_sidebands,
                           'NBkgSignalReg': n_bkg_signal_region, 'NBkgSignalRegErr': err_n_bkg_signal_region},
                          index=[0])
    del fit

    return result


for d_i in d_pt_bins:
    for e_i in e_pt_bins:
        print(e_i)
        print(d_i)

        particles_in_this_bin = pairs[(pairs['EPtBin'] == e_i) & (pairs['DPtBin'] == d_i)]
        results = correlation_dmeson(particles_in_this_bin, config_corr, d_pt_bin=d_i, e_pt_bin=e_i)
        plt.tight_layout()
        results['InvMassFit'].iloc[0].get_figure().savefig('mass_pt' + str(d_i) + '.pdf', bbox_inches="tight")


# plt.show()


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
