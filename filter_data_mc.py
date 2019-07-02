import dhfcorr.data_reader as dr
import dhfcorr.selection as sl


def root_to_h5(config_file='default_config_local.yaml'):
    path_to_data = "data/mc_root/"
    run_list = ['all']
    config = "mc_d_mesons"

    file = sl.CutsYaml(config_file)
    e_cuts = sl.Cuts(file, 'electron')
    d_cuts = sl.Cuts(file, 'D0')

    for run in run_list:
        print("Processing run: " + str(run))
        try:
            ele, d_meson = dr.read_root_file(path_to_data + str(run) + '.root', config)
            sl.build_add_features_dmeson(d_meson, d_cuts)
            dr.save(d_meson, config, "d_meson", run)
        except FileNotFoundError:
            print("Error to process run: " + str(run))


root_to_h5()
