import ROOT
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import root_numpy as rnp
import dhfcorr.correlate as corr
import dhfcorr.histogram as hist

sns.set()
# sns.set_palette('Reds_d')

ROOT.TH1.AddDirectory(False)

df_e = pd.read_hdf('filtered_e.hdf', 'electrons')
df_d = pd.read_hdf('filtered.hdf', 'D0')

config_corr = corr.CorrConfig()

# First create the pairs with all the selected D mesons and Electrons
# This is necessary because since both D meson and electron where selected.
# So we need to check if they are still in the same event.

df_d['IsSelectedD0'] = df_d['IsSelectedD0'] > 0.1
df_d['IsSelectedD0bar'] = df_d['IsSelectedD0bar'] > 0.1

df_d = df_d[df_d['IsSelectedD0'] | df_d['IsSelectedD0bar']]

pairs, selected_d, selected_e = corr.build_pairs(df_d, df_e, config_corr.correlation)

histo = hist.Histogram().from_dataframe(pairs, ('CentralityBin', 'VtxZBin', 'DPtBin',
                                                'EPtBin', 'DeltaEtaBin', 'DeltaPhiBin'))

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

        # histogram.SetTitle('d_pt:' + str(d_i) + ' e_pt:' + str(e_i))
        # histogram.Draw()
        # fit_mass.DrawHere(ROOT.gPad, 2)

        # Plot the invariant mass fit
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

        background_sidebands_2 = ROOT.Double()
        error_bkg_sidebands_2 = ROOT.Double()
        fit.Background(mean + 4. * sigma, mean + 10 * sigma, background_sidebands_2, error_bkg_sidebands_2)

        background_sidebands = background_sidebands_1 + background_sidebands_2
        error_bkg_sidebands = np.sqrt(error_bkg_sidebands_1 ** 2 + error_bkg_sidebands_2 ** 2)

        signal = ROOT.Double()
        err_signal = ROOT.Double()
        fit.Signal(n_sigma, signal, err_signal)

        # normalize the background correlation
        bkg_corr = bkg_corr / (background_sidebands / background)

        ax_bkg = bkg_corr.plot1d('DeltaPhiBin', color='black')
        signal_corr.plot1d('DeltaPhiBin', ax_bkg, color='red')

        if signal > 0:
            ax_sig = (signal_corr - bkg_corr).plot1d('DeltaPhiBin', color='blue')
            ax_sig.set_ylim(0, 1.1 * ax_sig.get_ylim()[1])

        correlation_signal_d.append(signal_corr)
        correlation_bkg_d.append(bkg_corr)

        # Problem in the automatic python bindings: the destructor is twice. Call it manually.
        del fit

    mass_fits_hist.append(list_e)
    correlation_signal.append(correlation_signal_d)
    correlation_bkg.append(correlation_bkg_d)

plt.show()
