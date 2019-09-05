import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import ROOT
import root_numpy as rnp
import seaborn as sns
import dhfcorr.io.data_reader as reader

sns.set()

import dhfcorr.config_yaml as configyaml
import dhfcorr.definitions as definitions
from dhfcorr.correlation_utils import reduce_to_single_particle

ROOT.TH1.AddDirectory(False)


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


def fit_inv_mass_root(histogram, range_fit, sig_func, bkg_func, fix_mean=None, fix_sigma=None, **kwargs):
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

    minimum = range_fit[0]
    maximum = range_fit[1]
    if minimum > maximum:
        raise ValueError('Minimum invariant mass is higher than maximum invariant mass. Check limits.')
    try:
        bkg_func = getattr(ROOT.AliHFInvMassFitter, bkg_func)
    except AttributeError as err:
        print(err)
        raise ValueError("Value of background function not found on AliHFInvMassFitter.")
    try:
        sig_func = getattr(ROOT.AliHFInvMassFitter, sig_func)
    except AttributeError as err:
        print(err)
        raise ValueError("Value of background function not found on AliHFInvMassFitter.")

    fit_mass = ROOT.AliHFInvMassFitter(histogram, minimum, maximum, bkg_func, sig_func)

    if fix_mean is not None:
        if not isinstance(fix_mean, float):
            raise TypeError('The value to fix the mean should be a float')
        fit_mass.SetFixGaussianMean(fix_mean)

    if fix_sigma is not None:
        if not isinstance(fix_sigma, float):
            raise TypeError('The value to fix the standard deviation should be a float')
        fit_mass.SetFixGaussianSigma(fix_sigma)

    fit_result = fit_mass.MassFitter(False)

    if fit_result == 0:  # fit failed
        raise RuntimeError("Fit has failed")

    return fit_mass


def make_histo_and_fit_inv_mass(df, n_bins='auto', min_hist=1.7, max_hist=2.1, suffix='', **kwargs):
    inv_mass = df['InvMass' + suffix]
    weight = df['Weight' + suffix]

    if isinstance(n_bins, str):
        bins = np.array(np.histogram_bin_edges(inv_mass, bins=n_bins, range=(min_hist, max_hist)), dtype='float64')
        histogram = ROOT.TH1F("MassFit", "MassFit", 2 * (len(bins) - 1), min_hist, max_hist)
    else:
        histogram = ROOT.TH1F("MassFit", "MassFit", n_bins, min_hist, max_hist)

    histogram.Sumw2()
    rnp.fill_hist(histogram, inv_mass, weights=weight)

    fit = fit_inv_mass_root(histogram, **kwargs)
    return fit


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


def fit_inv_mass(df, suffix, **kwargs):
    d_in_this_pt = reduce_to_single_particle(df, suffix)
    return make_histo_and_fit_inv_mass(d_in_this_pt, suffix=suffix, **kwargs)


def fit_d_meson_inv_mass(config_file_name=None, suffix='_t'):
    config = configyaml.ConfigYaml(config_file_name)
    data_sample = reader.load_pairs(config_file_name, 'selected')
    base_folder = config.values['base_folder']
    print("Fitting the Invariant Mass")
    fits = data_sample.groupby(['APtBin', 'TPtBin']).apply(fit_inv_mass, suffix=suffix, **config.values['inv_mass'])

    fits.columns = ['Fits']
    fits.to_pickle(base_folder + '/fits_inv_mass' + suffix + '.pkl')

    print("Plotting the fits")
    for index, row in fits.iteritems():
        a_i = index[0]
        t_i = index[1]
        fig, ax = plt.subplots()

        plot_inv_mass_fit(row, ax, **config.values['correlation_qa_style'])
        fig.savefig(base_folder + '/plots/mass_pt_a' + str(a_i) + '_t' + str(t_i) + '.pdf', bbox_inches="tight")


if __name__ == '__main__':
    """"Fit the invariant mass of the pairs.
    The first argument is the name of the file that contain the pairs of D-e. If none is provided, the default name 
    from definitions.py is used.
    The second argument is the name of the configuration yaml file. If none is provided, the default one is used.
    """

    import sys

    try:
        file_name = str(sys.argv[1])
    except IndexError:
        file_name = definitions.pairs

    try:
        config_file = sys.argv[2]
    except IndexError:
        config_file = None

    fit_d_meson_inv_mass(config_file)
