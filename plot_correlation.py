import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()
sns.set_context('notebook')
sns.set_palette('Set1')
base_folder = 'hmv0'


def plot_correlation(base_folder='hmv0', name=''):
    result = pd.read_pickle(base_folder + '_results_correlation' + name + '.pkl')
    for mass_fit, e_i, d_i in zip(result['InvMassFit'], result['APtBin'], result['TPtBin']):
        fig = mass_fit.get_figure()
        fig.savefig(base_folder + '/mass_pt_e' + str(e_i) + '_d' + str(d_i) + name + '.pdf', bbox_inches="tight")

    for corr_d, corr_d_uls, corr_d_ls, e_i, d_i in zip(result['SignalRegCorrInc'], result['SignalRegCorrULS'],
                                                       result['SignalRegCorrLS'], result['APtBin'], result['TPtBin']):
        ax_sig = corr_d.project('DeltaPhiBin').plot1d('DeltaPhiBin', label='D - inclusive electrons')
        corr_d_uls.project('DeltaPhiBin').plot1d('DeltaPhiBin', label='D - ULS electrons', ax=ax_sig)
        corr_d_ls.project('DeltaPhiBin').plot1d('DeltaPhiBin', label='D - LS electrons', ax=ax_sig)

        ax_sig.legend()
        ax_sig.set_ylim(min([ax_sig.get_ylim()[0], 0]), 1.2 * ax_sig.get_ylim()[1])
        ax_sig.set_xlabel(r'$\Delta\varphi$ [rad]')
        ax_sig.set_ylabel(r'$N^{D-e}$')
        title = '$p_T^D = $' + str(d_i)
        ax_sig.set_title(title)
        ax_sig.get_figure().savefig(
            base_folder + '/correlation_pt_signal_inc_e' + str(e_i) + '_d' + str(d_i) + name + '.pdf',
            bbox_inches="tight")

    for corr_d, corr_d_uls, corr_d_ls, e_i, d_i in zip(result['BkgRegCorrInc'], result['BkgRegCorrULS'],
                                                       result['BkgRegCorrLS'], result['APtBin'], result['TPtBin']):
        ax_sig = corr_d.project('DeltaPhiBin').plot1d('DeltaPhiBin', label='Bkg - inclusive electrons')
        corr_d_uls.project('DeltaPhiBin').plot1d('DeltaPhiBin', label='Bkg - ULS electrons', ax=ax_sig)
        corr_d_ls.project('DeltaPhiBin').plot1d('DeltaPhiBin', label='Bkg - LS electrons', ax=ax_sig)

        ax_sig.legend()
        ax_sig.set_ylim(min([ax_sig.get_ylim()[0], 0]), 1.2 * ax_sig.get_ylim()[1])
        ax_sig.set_xlabel(r'$\Delta\varphi$ [rad]')
        ax_sig.set_ylabel('$N^{Bkg-e}$')
        title = '$p_T^D = $' + str(d_i)
        ax_sig.set_title(title)
        ax_sig.get_figure().savefig(
            base_folder + '/correlation_pt_bkg_e' + str(e_i) + '_d' + str(d_i) + name + '.pdf',
            bbox_inches="tight")

    for corr_d, e_i, d_i in zip(result['DMesonCorr'], result['APtBin'], result['TPtBin']):
        corr_d = corr_d.project('DeltaPhiBin')
        bins = corr_d.get_bins('DeltaPhiBin')
        bin_size = bins[4] - bins[3]
        corr_d = corr_d / bin_size
        ax_sig = corr_d.plot1d('DeltaPhiBin', label='D - semi inclusive electrons')
        ax_sig.legend()
        ax_sig.set_ylim(min([ax_sig.get_ylim()[0], 0]), 1.2 * ax_sig.get_ylim()[1])
        title = '$p_T^D = $' + str(d_i)
        ax_sig.set_title(title)
        ax_sig.set_xlabel(r'$\Delta\varphi$ [rad]')
        ax_sig.set_ylabel(r'$\frac{1}{N^{D^0}} \frac{dN}{d\Delta\varphi}$')
        ax_sig.get_figure().savefig(base_folder + '/correlation_pte' + str(e_i) + '_d' + str(d_i) + name + '.pdf',
                                    bbox_inches="tight")

    for bkg_corr, signal_corr, e_i, d_i in zip(result['BkgRegCorr'], result['SignalRegCorr'], result['APtBin'],
                                               result['TPtBin']):
        ax_bkg = bkg_corr.project('DeltaPhiBin').plot1d('DeltaPhiBin', label='Background (normalized to signal region)')
        signal_corr.project('DeltaPhiBin').plot1d('DeltaPhiBin', ax_bkg, label='Signal Region')
        ax_bkg.set_ylim(min([ax_bkg.get_ylim()[0], 0]), 1.2 * ax_bkg.get_ylim()[1])
        ax_bkg.set_xlabel(r'$\Delta\varphi$ [rad]')
        ax_bkg.set_ylabel('$N_{D-e}$')
        ax_bkg.legend()
        title = '$p_T^D = $' + str(d_i)
        ax_bkg.set_title(title)
        ax_bkg.get_figure().savefig(
            base_folder + '/correlation_intermediate_pte' + str(e_i) + '_d' + str(d_i) + name + '.pdf',
            bbox_inches="tight")

    for bkg_corr, signal_corr, e_i, d_i in zip(result['BkgRegCorrMix'], result['SignalRegCorrMix'], result['APtBin'],
                                               result['TPtBin']):
        ax_bkg = (bkg_corr.project('DeltaEtaBin') / bkg_corr.get_n_bins('DeltaPhiBin')) \
            .plot1d('DeltaEtaBin', label='Background (normalized to signal region)')
        (signal_corr.project('DeltaEtaBin') / signal_corr.get_n_bins('DeltaPhiBin')) \
            .plot1d('DeltaEtaBin', ax_bkg, label='Signal Region')
        ax_bkg.legend()

        ax_bkg.set_xlabel(r'$\Delta\varphi$ [rad]')
        ax_bkg.set_ylabel('$N_{D-e}$')

        title = '$p_T^D = $' + str(d_i)
        ax_bkg.set_title(title)
        ax_bkg.set_ylim(min([ax_bkg.get_ylim()[0], 0]), 1.2 * ax_bkg.get_ylim()[1])
        ax_bkg.get_figure().savefig(base_folder + '/mix_eta_pte' + str(e_i) + '_d' + str(d_i) + name + '.pdf',
                                    bbox_inches="tight")


names = ['_inc']

for name in names:
    plot_correlation('hmv0', name)
