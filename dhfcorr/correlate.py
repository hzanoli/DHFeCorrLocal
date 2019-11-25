import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

from dhfcorr.correlation_utils import reduce_to_single_particle, compute_angular_differences
from dhfcorr.fit_inv_mass import select_inv_mass
from dhfcorr.fit.fit1D import plot_inv_mass_fit, make_histo_and_fit_inv_mass, get_n_signal, get_n_bkg, get_significance, \
    get_n_bkg_sidebands

from histogramming.histogram import Histogram
import sklearn.utils as skutils
import dhfcorr.io.data_reader as reader

try:
    import ROOT

    ROOT.TH1.AddDirectory(False)
    import root_numpy as rnp
except ImportError as error:
    print(error)
    warnings.warn('ROOT is not available. The functionality might be limited. Please check setup.')
    ROOT = None
    rnp = None


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
                       plot=False, identifier=('GridPID', 'EventNumber'), mix=True, subtract_non_hfe=False,
                       **kwargs):
    d_in_this_pt = reduce_to_single_particle(df_pairs, suffixes[0])
    e_in_this_pt = reduce_to_single_particle(df_pairs, suffixes[1])

    # print(df_pairs.name)dd
    # print(fits)
    # fit = fits.loc[df_pairs.name]
    # Fit Invariant Mass
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
                                     is_mixed=True, **kwargs['correlation'])
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


def prepare_single_particle_df(df, suffix, bins):
    """"Preprocessor before calculating the pairs. Takes place 'inplace' (changes df).
    Changes the names of the columns by appending the suffix.
    Adds values for weights in case they are not available.

    Returns the value of the columns before the names were changed and the new values.
    """

    # Add possibility to have weights in the correlations. If now available, create with 1.0
    if 'Weight' not in df.columns:
        df['Weight'] = 1.0
        df['Weight'] = df['Weight'].astype(np.float32)

    # Create the bins for each particle
    prefix = suffix[1:].upper()
    df['PtBin'] = pd.cut(df['Pt'], bins)


def fill_missing_value(correlation, value_name, suffixes, bins_value, new_value=0.0):
    """"Check if value_name is defined for trigger or assoc.
    Copies the value to the one that are not defined.If the feature mentioned in value_name is not found,
    creates a new one with values new_value. The bin is set to 0."""

    if value_name + suffixes[0] in correlation.columns:
        correlation[value_name + 'Bin'] = pd.cut(correlation[value_name + suffixes[0]], bins_value)
        correlation[value_name + suffixes[1]] = correlation[value_name + suffixes[0]]

    elif value_name + suffixes[1] in correlation.columns:
        correlation[value_name + 'Bin'] = pd.cut(correlation[value_name + suffixes[1]], bins_value)
        correlation[value_name + suffixes[0]] = correlation[value_name + suffixes[1]]
    else:
        # if no value available, save all of them to one on the bin new_value
        correlation[value_name + suffixes[0]] = new_value
        correlation[value_name + suffixes[1]] = new_value
        correlation[value_name + 'Bin'] = pd.cut(correlation[value_name + suffixes[1]], bins_value)


def build_pairs(trigger, associated, suffixes=('_d', '_e'), identifier=('GridPID', 'EventNumber'), is_mixed=False,
                n_to_mix=100, remove_same_id=False, chunk_size=10000, **kwargs):
    """"Builds a DataFrame with pairs of trigger and associated particles.
    This should always be the first step in the analysis.
    It assures that all trigger and associated particles are in the same event.
    This could have been lost since selections were applied on each of them.

    Returns a dataframe with the pairs

    Parameters
    ----------
    trigger : pd.DataFrame
        DataFrame with the trigger particles
    associated : pd.DataFrame
        DataFrame with associated particles
    suffixes: tuple
        suffixes are (in order) the values which will be used to name the trigger and associated particles
    identifier: tuple
        Column use to identify the particles. Should have be present in both trigger and associated.
    is_mixed: bool
        flag for mixing event analysis (the triggers are combined with associated from different events
    n_to_mix: int
        number of combinations used in the mixing. Only valid if is_mixed=True

    kwargs : dict
        Information used to build the correlation function

    Returns
    -------
    correlation: pd.Dataframe
        A DataFrame with the information of trigger and associated particles. The angular differences in phi and eta are
        also calculated in the columns DeltaPhi, DeltaEta (binned in DeltaPhiBin and DeltaEtaBin)

    Raises
    ------

    """

    # Type check
    if not isinstance(trigger, pd.DataFrame):
        raise TypeError('Value passed for trigger is not a DataFrame')
    if not isinstance(associated, pd.DataFrame):
        raise TypeError('Value passed for assoc is not a DataFrame')

    if isinstance(identifier, (str, float)):
        identifier = tuple([identifier])

    if is_mixed:
        # divide in chunks
        print("Mixing")
        trig_mix = pd.concat([trigger.reset_index()] * n_to_mix, ignore_index=True)
        assoc_mix = pd.concat(skutils.shuffle([associated.reset_index()] * n_to_mix), ignore_index=True)
        correlation = trig_mix.join(assoc_mix, lsuffix=suffixes[0], rsuffix=suffixes[1])
        del trig_mix, assoc_mix

        is_same_event = correlation[correlation.columns[0]] == correlation[correlation.columns[0]]  # set all to true
        for col in identifier:
            is_same_event = is_same_event & (correlation[col + suffixes[0]] == correlation[col + suffixes[1]])

        correlation = correlation[~is_same_event]
        print("End Mixing")

    else:  # Same Event
        # This needs len(trigger) x len(associated) to fit in memory
        correlation = trigger.join(associated, lsuffix=suffixes[0], rsuffix=suffixes[1], how='inner')

        if remove_same_id:
            correlation = correlation[correlation['ID' + suffixes[0]] != correlation['ID' + suffixes[1]]]

    # fill_missing_value(correlation, 'Centrality', suffixes, kwargs['bins_cent'], 1.0)
    # fill_missing_value(correlation, 'VtxZ', suffixes, kwargs['bins_zvtx'], 1.0)

    # Calculate the angular differences
    compute_angular_differences(correlation, suffixes=suffixes, **kwargs)

    return correlation


def build_pairs_from_lazy(df, suffixes, pt_bins_trig, pt_bins_assoc, filter_trig=None, filter_assoc=None, **kwargs):
    trig_suffix = suffixes[0]
    assoc_suffix = suffixes[1]

    sum_pairs = list()

    for trig, assoc in df:
        trig_df = trig.load()
        assoc_df = assoc.load()

        print(reader.get_friendly_parquet_file_name(trig.file_name)[:-6])

        prepare_single_particle_df(trig_df, trig_suffix, pt_bins_trig)
        prepare_single_particle_df(assoc_df, assoc_suffix, pt_bins_assoc)

        if filter_trig is not None:
            trig_df = trig_df.groupby('PtBin', as_index=False).apply(filter_trig)
        if filter_assoc is not None:
            assoc_df = assoc_df.groupby('PtBin', as_index=False).apply(filter_assoc)

        pairs = build_pairs(trig_df, assoc_df, (trig_suffix, assoc_suffix), **kwargs)
        sum_pairs.append(pairs)

    return pd.concat(sum_pairs)


if __name__ == '__main__':
    pass
