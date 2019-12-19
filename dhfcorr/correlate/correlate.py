import warnings

import matplotlib.pyplot as plt
import pandas as pd
from histogramming.histogram import Histogram

from dhfcorr.correlate.correlation_utils import reduce_to_single_particle
from dhfcorr.correlate.make_pairs import build_pairs
from dhfcorr.fit.fit1D import plot_inv_mass_fit, make_histo_and_fit_inv_mass, get_n_signal, get_n_bkg, get_significance, \
    get_n_bkg_sidebands
from dhfcorr.fit_inv_mass import select_inv_mass


def correlation_signal_region(df, fit, n_sigma, suffix='_d', axis=()):
    mean, sigma = (fit.GetMean(), fit.GetSigma())

    signal = select_inv_mass(df, mean - n_sigma * sigma, mean + n_sigma * sigma, suffix=suffix)

    signal_corr = Histogram.from_dataframe(signal, axis)
    return signal_corr


def correlation_background_region(df, fit, n_sigma_min, n_sigma_max, suffix='_d', axis=()):
    mean, sigma = (fit.GetMean(), fit.GetSigma())

    right = select_inv_mass(df, mean + n_sigma_min * sigma, mean + n_sigma_max * sigma, suffix=suffix)
    left = select_inv_mass(df, mean - n_sigma_max * sigma, mean - n_sigma_min * sigma, suffix=suffix)

    bkg = pd.concat([right, left])
    bkg_corr = Histogram.from_dataframe(bkg, axis)
    return bkg_corr


def normalize_mixed_event(histogram):
    p1 = histogram.data['Content'].loc[0. + 0.0001, 0. + 0.0001]
    # p2 = histogram.data['Content'].loc[0. + 0.0001, 0. - 0.0001]
    p2 = p4 = 0.
    p3 = histogram.data['Content'].loc[0. - 0.0001, 0. + 0.0001]
    # p4 = histogram.data['Content'].loc[0. - 0.0001, 0. - 0.0001]

    # TODO: Add finite bin correction
    counts_at_0 = (p1 + p2 + p3 + p4) / 2.
    return histogram / counts_at_0


def correlation_dmeson(df_pairs, n_sigma_sig=2., n_sigma_bkg_min=4., n_sigma_bkg_max=8.,
                       suffixes=('_d', '_e'),
                       axis=('CentralityBin', 'VtxZBin', 'DPtBin', 'EPtBin', 'DeltaEtaBin', 'DeltaPhiBin'),
                       plot=False, identifier=('RunNumber', 'EventNumber'), mix=True, subtract_non_hfe=False,
                       **kwargs):
    d_in_this_pt = reduce_to_single_particle(df_pairs, suffixes[0])
    e_in_this_pt = reduce_to_single_particle(df_pairs, suffixes[1])

    try:
        fit = make_histo_and_fit_inv_mass(d_in_this_pt, suffix=suffixes[0], **kwargs['inv_mass'])
    except RuntimeError:
        warnings.warn("Skipping value due to fit failure.")
        return pd.DataFrame(None, columns=['InvMassFit', 'DMesonCorr', 'SignalRegCorr', 'NSignal', 'NSignalErr',
                                           'SignalRegCorrMix', 'BkgRegCorr', 'NBkgSideBands', 'NBkgSideBandsErr',
                                           'BkgRegCorrMix', 'NBkgSignalReg', 'NBkgSignalRegErr', 'Significance',
                                           'Significance error'])

    n_signal, err_n_signal = get_n_signal(fit, n_sigma_sig)
    n_bkg_sidebands, err_n_bkg_sb = get_n_bkg_sidebands(fit, n_sigma_bkg_min, n_sigma_bkg_max)
    n_bkg_signal_region, err_n_bkg_signal_region = get_n_bkg(fit, n_sigma_sig)

    # Same Event

    # Inclusive electron
    signal_corr_inc = correlation_signal_region(df_pairs, fit, n_sigma_sig,
                                                axis=['DeltaEtaBin', 'DeltaPhiBin'], suffix=suffixes[0])
    bkg_corr_inc = correlation_background_region(df_pairs, fit, n_sigma_bkg_min,
                                                 n_sigma_bkg_max, suffix=suffixes[0],
                                                 axis=['DeltaEtaBin', 'DeltaPhiBin'])

    if subtract_non_hfe:
        # ULS electrons
        signal_corr_uls = correlation_signal_region(df_pairs.loc[df_pairs['NULS' + suffixes[1]] > 0], fit, n_sigma_sig,
                                                    axis=['DeltaEtaBin', 'DeltaPhiBin'], suffix=suffixes[0])
        bkg_corr_uls = correlation_background_region(df_pairs.loc[df_pairs['NULS' + suffixes[1]] > 0], fit,
                                                     n_sigma_bkg_min, n_sigma_bkg_max, suffix=suffixes[0],
                                                     axis=['DeltaEtaBin', 'DeltaPhiBin'])

        # LS electrons
        signal_corr_ls = correlation_signal_region(df_pairs.loc[df_pairs['NLS' + suffixes[1]] > 0], fit, n_sigma_sig,
                                                   axis=['DeltaEtaBin', 'DeltaPhiBin'], suffix=suffixes[0])

        bkg_corr_ls = correlation_background_region(df_pairs.loc[df_pairs['NLS' + suffixes[1]] > 0], fit,
                                                    n_sigma_bkg_min,
                                                    n_sigma_bkg_max, suffix=suffixes[0],
                                                    axis=['DeltaEtaBin', 'DeltaPhiBin'])

        signal_corr = signal_corr_inc - signal_corr_uls + signal_corr_ls
        bkg_corr = bkg_corr_inc - bkg_corr_uls + bkg_corr_ls

    else:
        signal_corr_uls = Histogram()
        signal_corr_ls = Histogram()
        bkg_corr_uls = Histogram()
        bkg_corr_ls = Histogram()

        signal_corr = signal_corr_inc
        bkg_corr = bkg_corr_inc

    if mix:
        # Mixed event
        df_mixed_pairs = build_pairs(d_in_this_pt, e_in_this_pt, suffixes=suffixes, identifier=identifier,
                                     is_mixed=True)
        signal_corr_mix = correlation_signal_region(df_mixed_pairs, fit, n_sigma_sig, suffix=suffixes[0],
                                                    axis=['DeltaEtaBin', 'DeltaPhiBin'])
        bkg_corr_mix = correlation_background_region(df_mixed_pairs, fit, n_sigma_bkg_min, n_sigma_bkg_max,
                                                     axis=['DeltaEtaBin', 'DeltaPhiBin'], suffix=suffixes[0])
        # Division by M(0,0)
        signal_corr_mix = normalize_mixed_event(signal_corr_mix)
        bkg_corr_mix = normalize_mixed_event(bkg_corr_mix)

        # Same/mixed
        corrected_signal_corr = (signal_corr / signal_corr_mix).project('DeltaPhiBin')

        corrected_bkg_corr = (bkg_corr / bkg_corr_mix).project('DeltaPhiBin')
        corrected_bkg_corr = corrected_bkg_corr / n_bkg_sidebands * n_bkg_signal_region

    else:
        signal_corr_mix = Histogram()
        bkg_corr_mix = Histogram()
        corrected_signal_corr = signal_corr.project('DeltaPhiBin')
        corrected_bkg_corr = bkg_corr.project('DeltaPhiBin')

    # normalize the background correlation
    bkg_corr = bkg_corr / n_bkg_sidebands * n_bkg_signal_region

    if n_signal > 0:
        d_meson_corr = (corrected_signal_corr - corrected_bkg_corr) / n_signal
    else:
        d_meson_corr = Histogram()

    if plot:
        fig, ax = plt.subplots()
        plot_inv_mass_fit(fit, ax, **kwargs['correlation_qa_style'])
    else:
        ax = None

    signif, err_signif = get_significance(fit, n_sigma_sig)

    result = pd.DataFrame({'InvMassFit': ax, 'DMesonCorr': d_meson_corr,
                           'SignalRegCorr': signal_corr, 'NSignal': n_signal, 'NSignalErr': err_n_signal,
                           'SignalRegCorrMix': signal_corr_mix,
                           'BkgRegCorr': bkg_corr, 'NBkgSideBands': n_bkg_sidebands, 'NBkgSideBandsErr': err_n_bkg_sb,
                           'BkgRegCorrMix': bkg_corr_mix,
                           'NBkgSignalReg': n_bkg_signal_region, 'NBkgSignalRegErr': err_n_bkg_signal_region,
                           'Significance': signif, 'Significance error': err_signif,
                           'SignalRegCorrInc': signal_corr_inc,
                           'SignalRegCorrULS': signal_corr_uls, 'SignalRegCorrLS': signal_corr_ls,
                           'BkgRegCorrInc': bkg_corr_inc,
                           'BkgRegCorrULS': bkg_corr_uls, 'BkgRegCorrLS': bkg_corr_ls
                           },
                          index=[0])

    # Problem in the automatic python bindings: the destructor is called twice. Calling it manually fix it.
    del fit
    return result


