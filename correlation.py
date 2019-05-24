import ROOT
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import root_numpy as rnp
import dhfcorr.correlate as corr

# sns.set(style="ticks", palette='bright')
sns.set()

ROOT.TH1.AddDirectory(False)

df_e = pd.read_hdf('filtered_e.hdf', 'electrons')
df_d = pd.read_hdf('filtered.hdf', 'D0')

df_d = df_d[df_d['Pt'] > 2.]

config_corr = corr.CorrConfig()

# First create the pairs with all the selected D mesons and Electrons
# This is necessary because since both D meson and electron where selected.
# So we need to check if they are still in the same event.

pairs, selected_d, selected_e = corr.build_pairs(df_d, df_e, config_corr.correlation)

histo = corr.create_histogram(pairs)

# Calculate the invariant mass for each (D meson, electron) pT bin
d_pt_bins = config_corr.correlation['bins_d']
e_pt_bins = config_corr.correlation['bins_e']

# Fit mass bins
mass_fits_hist = list()
mass_fit_obj = list()
canvas = ROOT.TCanvas("mass_plots", "mass_plots", 400 * (len(d_pt_bins) - 1), 300 * (len(e_pt_bins) - 1))
canvas.Divide(len(d_pt_bins) - 1, len(e_pt_bins) - 1)

# fig, ax = plt.subplots(len(e_pt_bins) - 1, len(d_pt_bins) - 1, squeeze=False,
#                       figsize=(8. * (len(d_pt_bins) - 1), 4. * (len(e_pt_bins) - 1)), constrained_layout=True)

# Fit invariant mass
fig_list = list()
df_list = list()
for d_i in range(len(d_pt_bins) - 1):
    list_e = list()
    mass_fits_e = list()
    df_list_e = list()
    for e_i in range(len(e_pt_bins) - 1):
        canvas.cd(e_i * (len(d_pt_bins) - 1) + d_i + 1)
        fig, ax = plt.subplots()
        fig_list.append(ax)
        # Get only pairs in this bin
        d_in_this_pt = selected_d[(selected_d['EPtBin'] == e_i) & (selected_d['DPtBin'] == d_i)]
        selected_mass, weights = corr.make_inv_mass(d_in_this_pt, 'D0')

        histogram = ROOT.TH1F("D0", "D0", 100, 1.3, 2.3)
        histogram.Sumw2()
        rnp.fill_hist(histogram, selected_mass, weights=weights)

        fit_mass = corr.fit_inv_mass_root(histogram, config_corr.correlation['inv_mass_lim'][d_i],
                                          config_corr.correlation['inv_mass_lim']['default'])

        histogram.SetTitle('d_pt:' + str(d_i) + ' e_pt:' + str(e_i))
        # histogram.Draw()
        fit_mass.DrawHere(ROOT.gPad, 2)
        # corr.plot_inv_mass_fit(fit_mass, ax[e_i][d_i], **config_corr.style_qa)
        corr.plot_inv_mass_fit(fit_mass, ax, **config_corr.style_qa)
        plt.tight_layout()

        mass_fits_e.append(fit_mass)
        list_e.append(histogram)

    mass_fit_obj.append(mass_fits_e)
    mass_fits_hist.append(list_e)

plt.show()

# Calculate the correlation for signal in each bin
correlation_signal = list()
for d_i in range(len(d_pt_bins) - 1):
    for e_i in range(len(e_pt_bins) - 1):
        fit = mass_fit_obj[d_i][e_i]
        particles_in_this_bin = pairs[(pairs['EPtBin'] == e_i) & (pairs['DPtBin'] == d_i)]

        signal = particles_in_this_bin[corr.select_inv_mass(particles_in_this_bin, 'D0', '_d')]
        print(signal)
