import pandas as pd
import seaborn as sns

sns.set()
sns.set_context('notebook')
sns.set_palette('Set1')
base_folder = 'cut_bdt'

result = pd.read_pickle(base_folder + '_results_correlation.pkl')

for mass_fit, e_i, d_i in zip(result['InvMassFit'], result['EPtBin'], result['DPtBin']):
    mass_fit.get_figure().savefig(base_folder + '/mass_pt_e' + str(e_i) + '_d' + str(d_i) + '.pdf', bbox_inches="tight")

for corr_d, e_i, d_i in zip(result['D    xMesonCorr'], result['EPtBin'], result['DPtBin']):
    ax_sig = corr_d.plot1d('DeltaPhiBin', label='D-Inclusive electron')
    ax_sig.set_ylim(min([ax_sig.get_ylim()[0], 0]), 1.2 * ax_sig.get_ylim()[1])
    ax_sig.legend()
    ax_sig.set_xlabel(r'$\Delta\varphi$ [rad]')
    ax_sig.set_ylabel(r'$\frac{1}{N^{D^0}} \frac{dN}{d\Delta\varphi}$')
    ax_sig.get_figure().savefig(base_folder + '/correlation_pte' + str(e_i) + '_d' + str(d_i) + '.pdf',
                                bbox_inches="tight")

for bkg_corr, signal_corr, e_i, d_i in zip(result['BkgRegCorr'], result['SignalRegCorr'], result['EPtBin'],
                                           result['DPtBin']):
    ax_bkg = bkg_corr.project('DeltaPhiBin').plot1d('DeltaPhiBin', label='Background (normalized to signal region)')
    signal_corr.project('DeltaPhiBin').plot1d('DeltaPhiBin', ax_bkg, label='Signal Region')
    ax_bkg.set_ylim(min([ax_bkg.get_ylim()[0], 0]), 1.2 * ax_bkg.get_ylim()[1])
    ax_bkg.legend()
    ax_bkg.get_figure().savefig(base_folder + '/correlation_intermediate_pte' + str(e_i) + '_d' + str(d_i) + '.pdf',
                                bbox_inches="tight")

for bkg_corr, signal_corr, e_i, d_i in zip(result['BkgRegCorrMix'], result['SignalRegCorrMix'], result['EPtBin'],
                                           result['DPtBin']):
    ax_bkg = (bkg_corr.project('DeltaEtaBin') / bkg_corr.get_n_bins('DeltaPhiBin')) \
        .plot1d('DeltaEtaBin', label='Background (normalized to signal region)')
    (signal_corr.project('DeltaEtaBin') / signal_corr.get_n_bins('DeltaPhiBin')) \
        .plot1d('DeltaEtaBin', ax_bkg, label='Signal Region')
    # ax_bkg.set_ylim(min([ax_bkg.get_ylim()[0], 0]), 1.2 * ax_bkg.get_ylim()[1])
    ax_bkg.legend()
    ax_bkg.get_figure().savefig(base_folder + '/mix_eta_pte' + str(e_i) + '_d' + str(d_i) + '.pdf', bbox_inches="tight")

for bkg_corr, signal_corr, e_i, d_i in zip(result['BkgRegCorrMix'], result['SignalRegCorrMix'], result['EPtBin'],
                                           result['DPtBin']):
    ax_bkg = (bkg_corr.project('DeltaPhiBin') / bkg_corr.get_n_bins('DeltaEtaBin')) \
        .plot1d('DeltaPhiBin', label='Background (normalized to signal region)')
    (signal_corr.project('DeltaPhiBin') / signal_corr.get_n_bins('DeltaEtaBin')) \
        .plot1d('DeltaPhiBin', ax_bkg, label='Signal Region')
    # ax_bkg.set_ylim(min([ax_bkg.get_ylim()[0], 0]), 1.2 * ax_bkg.get_ylim()[1])
    ax_bkg.legend()
    ax_bkg.get_figure().savefig(base_folder + '/mix_phi_pte' + str(e_i) + '_d' + str(d_i) + '.pdf', bbox_inches="tight")
