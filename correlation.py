import ROOT
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import root_numpy as rnp
import dhfcorr.correlate as corr
import histogram as hist

sns.set()
sns.set_context('notebook')
ROOT.TH1.AddDirectory(False)


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
        The same-event histogram for
    Raises
    ------

    """

    pairs, selected_d, selected_e = corr.build_pairs(df_d, df_e, suffixes, **config_corr.correlation)

    # Fit the Invariant mass for each pt_bin (in D and E)

    # selected_mass, weights

    prefixes = [suffix[1:].upper() for suffix in suffixes]

    d_bins = selected_d['PtBin' + prefixes[0]].cat.categories
    e_bins = selected_d['PtBin' + prefixes[1]].cat.categories






df_e = pd.read_hdf('filtered_e.hdf', 'electrons')
df_d = pd.read_hdf('filtered.hdf', 'D0')
# df_d = pd.read_parquet('selected.parquet')

config_corr = corr.CorrConfig()

# First create the pairs with all the selected D mesons and Electrons
# This is necessary because since both D meson and electron where selected.
# So we need to check if they are still in the same event.

pairs, selected_d, selected_e = corr.build_pairs(df_d, df_e, **config_corr.correlation)

# histo = hist.Histogram().from_dataframe(pairs, ('CentralityBin', 'VtxZBin',
# 'DPtBin','EPtBin', 'DeltaEtaBin', 'DeltaPhiBin'))

# Calculate the invariant mass for each (D meson, electron) pT bin
d_pt_bins = config_corr.correlation['bins_d']
e_pt_bins = config_corr.correlation['bins_e']

# Fit mass bins
mass_fits_hist = list()

# Fit invariant mass
fig_list = list()
df_list = list()
correlation_signal = list()
correlation_bkg = list()

for d_i in range(len(d_pt_bins) - 1):
    list_e = list()
    df_list_e = list()
    correlation_signal_d = list()
    correlation_bkg_d = list()

    for e_i in range(len(e_pt_bins) - 1):
        # Get only pairs in this bin
        d_in_this_pt = selected_d[(selected_d['EPtBin'] == e_i) & (selected_d['DPtBin'] == d_i)]
        selected_mass, weights = corr.make_inv_mass(d_in_this_pt, 'D0')

        histogram = ROOT.TH1F("D0", "D0", 100, 1.3, 2.3)
        histogram.Sumw2()
        rnp.fill_hist(histogram, selected_mass, weights=weights)

        fit = corr.fit_inv_mass_root(histogram, config_corr.correlation['inv_mass_lim'][d_i],
                                     config_corr.correlation['inv_mass_lim']['default'])

        # Plot the invariant mass fit_d_meson
        fig, ax = plt.subplots()
        corr.plot_inv_mass_fit(fit, ax, **config_corr.style_qa)
        plt.tight_layout()
        fig_list.append(ax)

        list_e.append(histogram)

        mean = fit.GetMean()
        sigma = fit.GetSigma()

        particles_in_this_bin = pairs[(pairs['EPtBin'] == e_i) & (pairs['DPtBin'] == d_i)]

        signal = corr.select_inv_mass(particles_in_this_bin, 'D0', mean - 2 * sigma, mean + 2 * sigma, suffix='_d')
        bkg = pd.concat([corr.select_inv_mass(particles_in_this_bin, 'D0', mean - 10. * sigma, mean - 4 * sigma,
                                              suffix='_d'),
                         corr.select_inv_mass(particles_in_this_bin, 'D0', mean + 4. * sigma, mean + 10 * sigma,
                                              suffix='_d')])

        n_sigma = 2.0
        background = ROOT.Double()
        error_bkg = ROOT.Double()
        fit.Background(n_sigma, background, error_bkg)

        axis = ('CentralityBin', 'VtxZBin', 'DPtBin', 'EPtBin', 'DeltaEtaBin', 'DeltaPhiBin')
        signal_corr = hist.Histogram.from_dataframe(signal, axis).project(['DeltaPhiBin'])
        bkg_corr = hist.Histogram.from_dataframe(bkg, axis).project(['DeltaPhiBin'])

        background_sidebands_1 = ROOT.Double()
        error_bkg_sidebands_1 = ROOT.Double()
        fit.Background(mean - 10. * sigma, mean - 4 * sigma, background_sidebands_1, error_bkg_sidebands_1)

        print('side bands 1 = ' + str(background_sidebands_1))

        background_sidebands_2 = ROOT.Double()
        error_bkg_sidebands_2 = ROOT.Double()
        fit.Background(mean + 4. * sigma, mean + 10 * sigma, background_sidebands_2, error_bkg_sidebands_2)

        print('side bands 2 = ' + str(background_sidebands_2))

        background_sidebands = background_sidebands_1 + background_sidebands_2
        error_bkg_sidebands = np.sqrt(error_bkg_sidebands_1 ** 2 + error_bkg_sidebands_2 ** 2)

        signal = ROOT.Double()
        err_signal = ROOT.Double()
        fit.Signal(n_sigma, signal, err_signal)

        # normalize the background correlation
        bkg_corr = bkg_corr / (background_sidebands / background)

        ax_bkg = bkg_corr.plot1d('DeltaPhiBin', color='black', label='Background (normalized to signal region)')
        signal_corr.plot1d('DeltaPhiBin', ax_bkg, color='red', label='Signal')
        ax_bkg.legend()
        ax_bkg.get_figure().savefig('correlation_intermediate_pt' + str(d_i) + '.pdf', bbox_inches="tight")

        if signal > 0:
            ax_sig = ((signal_corr - bkg_corr) / signal).plot1d('DeltaPhiBin', label='D-Inclusive electron')
            ax_sig.set_ylim(min([ax_sig.get_ylim()[0], 0]), 1.2 * ax_sig.get_ylim()[1])
            ax_sig.legend()
            ax_sig.set_xlabel(r'$\Delta\varphi$ [rad]')
            ax_sig.set_ylabel(r'$\frac{1}{N^{D^0}} \frac{dN}{d\Delta\varphi}$')
            ax_sig.get_figure().savefig('correlation_pt' + str(d_i) + '.pdf', bbox_inches="tight")

        correlation_signal_d.append(signal_corr)
        correlation_bkg_d.append(bkg_corr)

        fig.savefig('mass_pt' + str(d_i) + '.pdf', bbox_inches="tight")

        # Problem in the automatic python bindings: the destructor is twice. Call it manually.
        del fit

    mass_fits_hist.append(list_e)
    correlation_signal.append(correlation_signal_d)
    correlation_bkg.append(correlation_bkg_d)

plt.show()
