import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt

try:
    import ROOT
except ImportError as error:
    print(error)
    print('ROOT is not available. The functionality might be limited.')
    ROOT = None
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

    ax.errorbar(bins_center, content, yerr=y_error, xerr=np.array(bins_width) / 2., label='Data',
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
    bkg = ROOT.Double()
    error_bkg = ROOT.Double()
    fit.Background(n_sigma, bkg, error_bkg)

    signif = ROOT.Double()
    err_signif = ROOT.Double()
    fit.Significance(n_sigma, signif, err_signif)

    signal = ROOT.Double()
    err_signal = ROOT.Double()
    fit.Signal(n_sigma, signal, err_signal)

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


def build_pairs(trigger, associated, suffixes=('_d', '_e'), identifier=('GridPID', 'EventNumber'), **kwargs):
    """"Builds a DataFrame with pairs of trigger and associated particles.

    identifier should have be present in both trigger and associated.
    suffixes are (in order) the values which will be used to name the trigger and associated particles

    This should always be the first step in the analysis.
    It assures that all trigger and associated particles are in the same event.
    This could have been lost since selections were applied on each of them.

    Returns a dataframe with the pairs, other one with the triggers and another one with the associated
    It is more convenient to use the pairs to build correlations and the individual for normalizations (and for mixing).


    Parameters
    ----------
    trigger : pd.Dataframe
        DataFrame with the trigger particles
    associated : pd.Dataframe
        DataFrame with associated particles
    suffixes:
    identifier:
    kwargs : dict
        Information

    Returns
    -------

    Raises
    ------

    """

    # Type check
    if not isinstance(trigger, pd.DataFrame):
        raise TypeError('Value passed for trigger is not a DataFrame')
    if not isinstance(associated, pd.DataFrame):
        raise TypeError('Value passed for assoc is not a DataFrame')

    # Copy the DataFrames to avoid changing the original ones
    trigger = trigger.copy()
    associated = associated.copy()

    # Prepare the features and add possible missing ones.
    prepare_single_particle_df(trigger, suffixes[0], **kwargs)
    prepare_single_particle_df(associated, suffixes[1], **kwargs)

    # Build the correlation pairs
    feat_on_left = [str(x) + suffixes[0] for x in identifier]
    feat_on_right = [str(x) + suffixes[1] for x in identifier]
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
