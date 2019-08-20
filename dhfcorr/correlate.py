import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
import warnings
from histogram.histogram import Histogram
import sklearn.utils as skutils

try:
    import ROOT
    import root_numpy as rnp
except ImportError as error:
    print(error)
    warnings.warn('ROOT is not available. The functionality might be limited. Please check setup.')
    ROOT = None
    root_numpy = None
    pass


class CorrConfig(object):
    """Class to store the configuration used during the correlation"""

    def __init__(self, selection_file="../config/default_config_local.yaml", name="D0"):
        with open(selection_file, "r") as document:
            try:
                config = yaml.safe_load(document)
            except yaml.YAMLError as exc:
                print(exc)
                raise ValueError("Error when processing the file. Check the YAMLError above.")

        self.file = config
        self.correlation = self.file['correlation']
        self.style_qa = self.file['correlation_qa_style']


def plot_inv_mass_fit(fit, ax=None, **kwargs):
    """"Plots (using matplotlib) the results of the AliHFInvMassFitter.

   Parameters
   ----------


    fit: ROOT.AliHFInvMassFitter
        The fit_d_meson result that will be drawn

    ax : None or plt.axes
    **kwargs
        configuration of the plots.

   Returns
   -------
    ax: plt.axes
         Axes containing the plot.

   Raises
   ------
    ValueError
        If any of the parameters used to configure the plot is inconsistent.

    """
    if ax is None:
        fig, ax = plt.subplots()
    # Retrieve histogram values
    histogram = fit.GetHistoClone()
    content = [histogram.GetBinContent(i) for i in range(1, histogram.GetNbinsX() + 1)]  # bins go 1 -> N
    y_error = [histogram.GetBinError(i) for i in range(1, histogram.GetNbinsX() + 1)]
    bins_center = [histogram.GetXaxis().GetBinCenter(i) for i in range(1, histogram.GetNbinsX() + 1)]
    bins_width = [histogram.GetXaxis().GetBinWidth(i) for i in range(1, histogram.GetNbinsX() + 1)]

    # Plot hist

    if kwargs['x_range'][0] > kwargs['x_range'][1]:
        raise ValueError("Minimum value (x axis) in the mass plots is larger than the maximum value")

    ax.errorbar(bins_center, content, yerr=y_error, xerr=np.array(bins_width) / 2., markersize=3, label='Data',
                **kwargs['kwargs_plot'])
    ax.set_xlim(kwargs['x_range'])

    # Plot (total) fit_d_meson function
    x_values_func = np.linspace(kwargs['x_range'][0], kwargs['x_range'][1], 200)
    y_total_func = [fit.GetMassFunc().Eval(x) for x in x_values_func]
    y_bkg_func = [fit.GetBackgroundRecalcFunc().Eval(x) for x in x_values_func]
    ax.plot(x_values_func, y_total_func, label='Total Fit')
    ax.plot(x_values_func, y_bkg_func, label='Background Fit', linestyle='--')

    ax.set_xlabel('Invariant Mass [GeV/$c^2$]')
    ax.set_ylabel('Counts per {:.2f} MeV/$c^2$'.format(1000 * bins_width[0]))

    n_sigma = kwargs['n_sigma_significance']
    bkg, error_bkg = get_n_bkg(fit, n_sigma)
    signif, err_signif = get_significance(fit, n_sigma)
    signal, err_signal = get_n_signal(fit, n_sigma)

    text_to_plot_left = 'S ({0:.0f}$\\sigma$) = {1:.0f} $\\pm$ {2:.0f} \n'.format(n_sigma, signal, err_signal)
    text_to_plot_left += 'B ({0:.0f}$\\sigma$) = {1:.0f} $\\pm$ {2:.0f}'.format(n_sigma, bkg, error_bkg)
    text_to_plot_right = 'S/B ({0:.0f}$\\sigma$) = {1:.4f}\n'.format(n_sigma, signal / bkg)
    # TODO include reflections
    # pinfos->AddText(Form("Refl/Sig =  %.3f #pm %.3f ", fRflFunc->GetParameter(0), fRflFunc->GetParError(0)));
    text_to_plot_right += 'Significance({0:.0f}$\\sigma$) = {1:.1f} $\\pm$ {2:.1f}'.format(n_sigma, signif, err_signif)

    # a x.set_title(text_to_plot)
    ax.get_figure().subplots_adjust(top=0.7)
    ax.text(0.05, 1.125, text_to_plot_left, transform=ax.transAxes, verticalalignment='top')
    ax.text(0.55, 1.125, text_to_plot_right, transform=ax.transAxes, verticalalignment='top')

    text_chi2 = r"$\chi^2_{red}" + " = {0:1.2f}$".format(fit.GetReducedChiSquare())
    ax.plot(np.NaN, np.NaN, '-', color='none', label=text_chi2)

    text_mean = '$\\mu = ' + '{0:1.3f}'.format(fit.GetMean()) + '\\pm {0:2.3f}$'.format(fit.GetMeanUncertainty())
    text_mean += ' GeV/$c^2$'
    ax.plot(np.NaN, np.NaN, '-', color='none', label=text_mean)

    text_std = r'$\sigma = ' + '{0:2.1f}'.format(fit.GetSigma() * 1000.)
    text_std += '\\pm {0:2.1f}$'.format(fit.GetSigmaUncertainty() * 1000) + ' MeV/$c^2$'
    ax.plot(np.NaN, np.NaN, '-', color='none', label=text_std)

    handles, labels = plt.gca().get_legend_handles_labels()
    order = [5, 0, 1, 2, 3, 4]
    ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
              loc='best', frameon=False)

    return ax


def fit_inv_mass_root(histogram, config_inv_mass, fix_mean=None, fix_sigma=None):
    """"Fits the invariant mass distribution using AliHFInvMassFitter.

    Parameters
    ----------
    histogram : ROOT.TH1
        The histogram that will be fitted.
    config_inv_mass : dict
        Values used to configure the AliHFInvMassFitter. Should containt: range (the range that the fit_d_meson will be
        performed), bkg_func and sig_func(the function used to fit_d_meson the data, as defined in AliHFInvMassFitter.h)
    fix_mean: None or float
        In case it is not None, the fit_d_meson will fix the mean to this value.
    fix_sigma: None or float
        In case it is not None, the fit_d_meson will fix the standard deviation to this value.

    Returns
    -------
    fit_mass : ROOT.AliHFInvMassFitter
        The fit_d_meson mass object for this histogram

    Raises
    ------
    KeyError
        If the keywords (range, bkg_func, sig_func) used to configure the AliHFInvMassFitter are not found on
        config_inv_mass or config_inv_mass_def.
    ValueError
        In case the one of the configurations in config_inv_mass (or config_inv_mass_def) is not consistent.

    """

    # Copy dict to avoid changes

    local_dict = config_inv_mass.copy()
    try:
        minimum = local_dict['range'][0]
        maximum = local_dict['range'][1]
        if minimum > maximum:
            raise ValueError('Minimum invariant mass is higher than maximum invariant mass. Check limits.')
        try:
            bkg_func = getattr(ROOT.AliHFInvMassFitter, local_dict['bkg_func'])
        except AttributeError as err:
            print(err)
            raise ValueError("Value of background function not found on AliHFInvMassFitter.")
        try:
            sig_func = getattr(ROOT.AliHFInvMassFitter, local_dict['sig_func'])
        except AttributeError as err:
            print(err)
            raise ValueError("Value of background function not found on AliHFInvMassFitter.")

    except KeyError as kerr:
        print(kerr)
        print("Problem accessing the configuration of the mass fitter")
        raise

    fit_mass = ROOT.AliHFInvMassFitter(histogram, minimum, maximum, bkg_func, sig_func)

    if fix_mean is not None:
        if not isinstance(fix_mean, float):
            raise TypeError('The value to fix the mean should be a float')
        fit_mass.SetFixGaussianMean(fix_mean)

    if fix_sigma is not None:
        if not isinstance(fix_sigma, float):
            raise TypeError('The value to fix the standard deviation should be a float')
        fit_mass.SetFixGaussianSigma(fix_sigma)

    fit_mass.MassFitter(False)

    return fit_mass


def fit_d_meson_mass(df, n_bins='auto', min_hist=1.7, max_hist=2.1, **kwargs):
    inv_mass = df['InvMass']
    weight = df['Weight']

    if isinstance(n_bins, str):
        bins = np.array(np.histogram_bin_edges(inv_mass, bins=n_bins, range=(min_hist, max_hist)), dtype='float64')
        histogram = ROOT.TH1F("MassFit", "MassFit", 2 * (len(bins) - 1), min_hist, max_hist)
    else:
        histogram = ROOT.TH1F("MassFit", "MassFit", n_bins, min_hist, max_hist)

    histogram.Sumw2()
    rnp.fill_hist(histogram, inv_mass, weights=weight)

    fit = fit_inv_mass_root(histogram, kwargs['inv_mass_lim']['default'])
    return fit


def correlation_signal_region(df, fit, n_sigma, suffix='_d', axis=()):
    mean, sigma = (fit.GetMean(), fit.GetSigma())
    signal = select_inv_mass(df, mean - n_sigma * sigma, mean + n_sigma * sigma, suffix=suffix)
    signal_corr = Histogram.from_dataframe(signal, axis)

    return signal_corr


def correlation_background_region(df, fit, n_sigma_min, n_sigma_max, suffix='_d', axis=()):
    mean, sigma = (fit.GetMean(), fit.GetSigma())

    right = select_inv_mass(df, mean - n_sigma_max * sigma, mean - n_sigma_min * sigma, suffix=suffix)
    left = select_inv_mass(df, mean - n_sigma_max * sigma, mean - n_sigma_min * sigma, suffix=suffix)

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


def get_significance(fit, n_sigma):
    signif = ROOT.Double()
    err_signif = ROOT.Double()
    fit.Significance(n_sigma, signif, err_signif)
    return float(signif), float(err_signif)


def normalize_mixed_event(histogram):
    p1 = histogram.data['Content'].loc[0. + 0.0001, 0. + 0.0001]
    # p2 = histogram.data['Content'].loc[0. + 0.0001, 0. - 0.0001]
    p2 = p4 = 0.
    p3 = histogram.data['Content'].loc[0. - 0.0001, 0. + 0.0001]
    # p4 = histogram.data['Content'].loc[0. - 0.0001, 0. - 0.0001]

    # TODO: Add finite bin correction
    counts_at_0 = (p1 + p2 + p3 + p4) / 2.
    return histogram / counts_at_0


def correlation_dmeson(df_pairs, config_corr, n_sigma_sig=2., n_sigma_bkg_min=4., n_sigma_bkg_max=8.,
                       suffixes=('_d', '_e'),
                       axis=('CentralityBin', 'VtxZBin', 'DPtBin', 'EPtBin', 'DeltaEtaBin', 'DeltaPhiBin'),
                       plot=False, identifier=('GridPID', 'EventNumber')):
    d_in_this_pt = reduce_to_single_particle(df_pairs, suffixes[0])
    e_in_this_pt = reduce_to_single_particle(df_pairs, suffixes[1])

    # Fit Invariant Mass
    fit = fit_d_meson_mass(d_in_this_pt, **config_corr.correlation)

    n_signal, err_n_signal = get_n_signal(fit, n_sigma_sig)
    n_bkg_sidebands, err_n_bkg_sb = get_n_bkg_sidebands(fit, n_sigma_bkg_min, n_sigma_bkg_max)
    n_bkg_signal_region, err_n_bkg_signal_region = get_n_bkg(fit, n_sigma_sig)

    # Same Event
    signal_corr = correlation_signal_region(df_pairs, fit, n_sigma_sig, axis=['DeltaEtaBin', 'DeltaPhiBin'])
    bkg_corr = correlation_background_region(df_pairs, fit, n_sigma_bkg_min,
                                             n_sigma_bkg_max, axis=['DeltaEtaBin', 'DeltaPhiBin'])

    # Mixed event
    df_mixed_pairs = build_pairs(d_in_this_pt, e_in_this_pt, suffixes=suffixes, identifier=identifier, is_mixed=True,
                                 **config_corr.correlation)
    signal_corr_mix = correlation_signal_region(df_mixed_pairs, fit, n_sigma_sig,
                                                axis=['DeltaEtaBin', 'DeltaPhiBin'])
    bkg_corr_mix = correlation_background_region(df_mixed_pairs, fit, n_sigma_bkg_min, n_sigma_bkg_max,
                                                 axis=['DeltaEtaBin', 'DeltaPhiBin'])
    # Division by M(0,0)
    signal_corr_mix = normalize_mixed_event(signal_corr_mix)
    bkg_corr_mix = normalize_mixed_event(bkg_corr_mix)

    # Same/mixed
    corrected_signal_corr = signal_corr.project('DeltaPhiBin') / (
            signal_corr_mix.project('DeltaPhiBin') / signal_corr.get_n_bins('DeltaEtaBin'))

    # normalize the background correlation
    corrected_bkg_corr = bkg_corr.project('DeltaPhiBin') / (
            bkg_corr_mix.project('DeltaPhiBin') / bkg_corr.get_n_bins('DeltaEtaBin'))
    corrected_bkg_corr = corrected_bkg_corr / n_bkg_sidebands * n_bkg_signal_region
    bkg_corr = bkg_corr / n_bkg_sidebands * n_bkg_signal_region

    if n_signal > 0:
        d_meson_corr = (corrected_signal_corr - corrected_bkg_corr) / n_signal
    else:
        d_meson_corr = Histogram()

    if plot:
        fig, ax = plt.subplots()
        plot_inv_mass_fit(fit, ax, **config_corr.style_qa)
    else:
        ax = None

    signif, err_signif = get_significance(fit, n_sigma_sig)

    result = pd.DataFrame({'InvMassFit': ax, 'DMesonCorr': d_meson_corr,
                           'SignalRegCorr': signal_corr, 'NSignal': n_signal, 'NSignalErr': err_n_signal,
                           'SignalRegCorrMix': signal_corr_mix,
                           'BkgRegCorr': bkg_corr, 'NBkgSideBands': n_bkg_sidebands, 'NBkgSideBandsErr': err_n_bkg_sb,
                           'BkgRegCorrMix': bkg_corr_mix,
                           'NBkgSignalReg': n_bkg_signal_region, 'NBkgSignalRegErr': err_n_bkg_signal_region,
                           'Significance': signif, 'Significance error': err_signif},
                          index=[0])
    # Problem in the automatic python bindings: the destructor is called twice. Calling it manually fix it.
    del fit
    return result


def prepare_single_particle_df(df, suffix, **kwargs):
    """"Preprocessor before calculating the pairs. Takes place 'inplace' (changes df).
    Changes the names of the columns by appending the suffix.
    Adds values for weights in case they are not available.

    Returns the value of the columns before the names were changed and the new values.
    """

    # Get rid of the index coming from the Pt bins used to filter the trigger and/or d associated
    df.reset_index(drop=True, inplace=True)
    df['Id'] = df.index

    # Add possibility to have weights in the correlations. If now available, create with 1.0
    if 'Weight' not in df.columns:
        df['Weight'] = 1.0

    cols_original = df.columns
    df.columns = [str(x) + str(suffix) for x in df.columns]
    cols_new = df.columns

    # Create the bins for each particle
    prefix = suffix[1:].upper()
    df[prefix + 'PtBin'] = pd.cut(df['Pt' + suffix], kwargs['bins' + str(suffix)])

    return list(cols_original), list(cols_new)


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


def convert_to_range_pi2_3pi2(dphi):
    if dphi > 3. * np.pi / 2.:
        return dphi - 2. * np.pi
    if dphi < -np.pi / 2.:
        return dphi + 2. * np.pi
    return dphi


def convert_to_range(dphi):
    if dphi > 3. * np.pi / 2.:
        dphi = dphi - 2. * np.pi
    if dphi < -np.pi / 2.:
        dphi = dphi + 2. * np.pi

    if dphi < 0.:
        dphi = -dphi
    if dphi > np.pi:
        dphi = 2 * np.pi - dphi

    return dphi


def build_pairs(trigger, associated, suffixes=('_d', '_e'), identifier=('GridPID', 'EventNumber'), is_mixed=False,
                n_to_mix=500, **kwargs):
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

    # Copy the DataFrames to avoid changing the original ones
    trigger = trigger.copy()
    associated = associated.copy()

    # Prepare the features and add possible missing ones.
    prepare_single_particle_df(trigger, suffixes[0], **kwargs)
    prepare_single_particle_df(associated, suffixes[1], **kwargs)

    # Build the correlation pairs
    feat_on_left = [str(x) + suffixes[0] for x in identifier]
    feat_on_right = [str(x) + suffixes[1] for x in identifier]

    if is_mixed:
        trig_mix = pd.concat([trigger] * n_to_mix, ignore_index=True)
        assoc_mix = pd.concat(skutils.shuffle([associated] * n_to_mix), ignore_index=True)
        correlation = trig_mix.join(assoc_mix, rsuffix='x', lsuffix='y')
        del trig_mix, assoc_mix

        is_same_event = correlation[correlation.columns[0]] == correlation[correlation.columns[0]]  # set all to true

        for col in identifier:
            is_same_event = is_same_event & (correlation[col + suffixes[0]] == correlation[col + suffixes[1]])

        correlation = correlation[~is_same_event]

    else:  # Same Event
        correlation = trigger.merge(associated, left_on=feat_on_left, right_on=feat_on_right)

    fill_missing_value(correlation, 'Centrality', suffixes, kwargs['bins_cent'], 1.0)
    fill_missing_value(correlation, 'VtxZ', suffixes, kwargs['bins_zvtx'], 1.0)

    # Calculate the angular differences
    correlation['DeltaPhi'] = (correlation['Phi' + suffixes[0]] - correlation['Phi' + suffixes[1]]).apply(
        convert_to_range)
    correlation['DeltaEta'] = (correlation['Eta' + suffixes[0]] - correlation['Eta' + suffixes[1]])

    # Calculate the bins for angular quantities
    correlation['DeltaPhiBin'] = pd.cut(correlation['DeltaPhi'], kwargs['bins_phi'])
    correlation['DeltaEtaBin'] = pd.cut(correlation['DeltaEta'], kwargs['bins_eta'])

    # Calculate the weight of the pair
    correlation['Weight'] = correlation['Weight' + suffixes[0]] * correlation['Weight' + suffixes[1]]
    # Save the weight square that will be useful to calculate the errors
    correlation['WeightSquare'] = correlation['Weight'] ** 2

    return correlation


def reduce_to_single_particle(correlation, suffix):
    particle = correlation.groupby(by=['Id' + suffix]).nth(0).reset_index()
    cols_to_keep = [x for x in correlation.columns if x.endswith('Bin')]
    cols_to_keep += [x for x in correlation.columns if x.endswith(suffix)]
    particle = particle[cols_to_keep]
    particle.columns = [x[:-len(suffix)] if x.endswith(suffix) else x for x in particle.columns]

    return particle


def select_inv_mass(df, min_mass, max_mass, suffix=''):
    """"Select the rows of df dataframe which is its mass is between [min_mass, max_mass] and return a view.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with the D mesons. Should contain the column InvMass
    min_mass: float
        Minimum value that will be selected in the mass. Do not give it in number of sigmas.
    max_mass: float
        Maximum value that will be selected in the mass. Do not give it in number of sigmas.
    suffix: str
        Suffix used to modify the name of the DataFrame (e.g. '_d')
    Returns
    -------
    df_selected: pd.DataFrame
        Dataframe of the selected particles. Returns a view.

    """

    selected = ((df['InvMass' + suffix] >= min_mass) & (df['InvMass' + suffix] <= max_mass))  # Check max_mass
    return df[selected]


def mix(trigger, assoc_pool):
    print(trigger.name)
    selected_assoc = assoc_pool.fast_sample()
    for col in trigger.index:
        selected_assoc[col] = trigger.loc[col]
    return selected_assoc


class MixingSampler:
    """Class used to generate pseudo random data for the event mixing.
    It is not necessary to have independent random values for each trigger candidate, so it is much faster than using
    pd.DataFrame.sample.
    """

    def __init__(self, df, n_to_sample):
        self.data = df.copy()
        self.n_to_sample = n_to_sample
        self.n_iter = 0
        self.shuffle_counter = 0
        self.data_size = len(self.data)
        self.data.sample(frac=1.)

    def fast_sample(self):
        # Check if it is needed to shuffle. Skip this iteration since it does not fit in the data
        if int((self.n_iter * self.n_to_sample) / self.data_size) > self.shuffle_counter:
            self.data.sample(frac=1.)
            self.shuffle_counter += 1
            self.n_iter += 1

        start_idx = (self.n_iter * self.n_to_sample) % self.data_size
        self.n_iter += 1
        return self.data.iloc[start_idx:start_idx + self.n_to_sample]
