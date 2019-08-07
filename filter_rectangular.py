import dhfcorr.io.data_reader as dr
import dhfcorr.selection.selection as sl

dataset = dr.load('D0', 'd_meson')
selection_for_ptbin = sl.Cuts('cuts_jets.yaml')
selected = sl.filter_in_pt_bins(dataset, selection_for_ptbin)
selected.reset_index(drop=True).to_parquet('selected_rectangular.parquet')
