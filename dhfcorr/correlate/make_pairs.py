import numpy as np
import pandas as pd
from sklearn import utils as skutils

import dhfcorr.definitions as definitions
import dhfcorr.io.data_reader as reader
import dhfcorr.selection.selection as sl
from dhfcorr.config_yaml import ConfigYaml
from dhfcorr.correlate.correlation_utils import compute_angular_differences


def prepare_single_particle_df(df, bins, suffix=''):
    """"Preprocessor before calculating the pairs. Takes place 'inplace' (changes df).
    Changes the names of the columns by appending the suffix.
    Adds values for weights in case they are not available.

    Returns the value of the columns before the names were changed and the new values.
    """
    # Add possibility to have weights in the correlations. If now available, create with 1.0
    if 'Weight' not in df.columns:
        df['Weight'] = 1.0
        df['Weight'] = df['Weight'].astype(np.float32)

    # Create the bins for each particle
    df['PtBin'] = pd.cut(df['Pt'], bins)
    df['IdDF'] = df['PtBin'].reset_index().index

    # df.columns = [x + suffix for x in df.columns]


def build_pairs(trigger, associated, suffixes=('_t', '_a'), identifier=('RunNumber', 'EventNumber'), is_mixed=False,
                n_to_mix=100, remove_same_id=False):
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
    is_mixed: bool
        flag for mixing event analysis (the triggers are combined with associated from different events
    n_to_mix: int
        number of combinations used in the mixing. Only valid if is_mixed=True


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

    if is_mixed:
        # Repeat the trigger and assoc n times (should fit in memory
        trig_mix = pd.concat([trigger.reset_index()] * n_to_mix, ignore_index=True)
        assoc_mix = pd.concat(skutils.shuffle([associated.reset_index()] * n_to_mix), ignore_index=True)
        # Joins on the index, since it is faster
        correlation = trig_mix.join(assoc_mix, lsuffix=suffixes[0], rsuffix=suffixes[1])
        del trig_mix, assoc_mix

        # Remove particles from the same event
        is_same_event = correlation[correlation.columns[0]] == correlation[correlation.columns[0]]  # set all to true
        for col in identifier:
            is_same_event = is_same_event & (correlation[col + suffixes[0]] == correlation[col + suffixes[1]])

        correlation = correlation[~is_same_event]

    else:
        correlation = trigger.join(associated, lsuffix=suffixes[0], rsuffix=suffixes[1], how='inner')

        if remove_same_id:
            correlation = correlation[correlation['ID' + suffixes[0]] != correlation['ID' + suffixes[1]]]

    return correlation


def process_lazy_worker(df_list, suffixes, pt_bins_trig, pt_bins_assoc, filter_trig, filter_assoc,
                        remove_same_id=False):
    pairs_list = list()
    trig_suffix = suffixes[0]
    assoc_suffix = suffixes[1]
    for df in df_list:
        trig_df = df[0].load()
        prepare_single_particle_df(trig_df, pt_bins_trig, trig_suffix)

        if filter_trig is not None:
            trig_df = trig_df.groupby(['PtBin'], group_keys=False).apply(filter_trig)

        trig_df.drop('PtBin', axis='columns', inplace=True)

        if df[0] == df[1]:  # Saves half the time in case trigger and assoc are the same
            assoc_df = trig_df
        else:
            assoc_df = df[1].load()
            prepare_single_particle_df(assoc_df, pt_bins_assoc, assoc_suffix)

            if filter_assoc is not None:
                assoc_df = assoc_df.groupby(['PtBin'], group_keys=False).apply(filter_assoc)
            assoc_df.drop('PtBin', axis='columns', inplace=True)

        if trig_df.empty or assoc_df.empty:
            return None

        pairs = build_pairs(trig_df, assoc_df, (trig_suffix, assoc_suffix), remove_same_id=remove_same_id)
        pairs_list.append(pairs)

    return pd.concat(pairs_list)


def build_pairs_from_lazy(df_list, suffixes, pt_bins_trig, pt_bins_assoc, filter_trig=None, filter_assoc=None,
                          nthreads=1, remove_same_id=False):
    from dhfcorr.multiprocessing import process_multicore

    if nthreads == 1:
        correlation = process_lazy_worker(df_list, suffixes, pt_bins_trig, pt_bins_assoc, filter_trig, filter_assoc,
                                          remove_same_id)
    else:
        sum_pairs = process_multicore(
            lambda x: process_lazy_worker(df_list, suffixes, pt_bins_trig, pt_bins_assoc, filter_trig, filter_assoc,
                                          remove_same_id),
            df_list, nthreads, 'Paring')
        correlation = pd.concat(sum_pairs).drop()

    compute_angular_differences(correlation, suffixes=suffixes)

    return correlation


def make_pairs(file_list, id, config=None, is_dd=True):
    config_name = reader.get_dataset_name_from_file(file_list[0][0])
    folder_to_save = definitions.PROCESSING_FOLDER + config_name + '/pairs/'
    file_to_save = folder_to_save
    if is_dd:
        file_to_save += 'dd'
    else:
        file_to_save += 'de'
    file_to_save += '_pairs_' + str(id) + '.parquet'

    if isinstance(file_list, str):  # it will be loaded from a file
        file_list = pd.read_csv(file_list).values.tolist()

    config_yaml = ConfigYaml(config)

    cols_trig = config_yaml.values['pre_filter_ml']['cols_to_keep']
    pt_bins_trig = config_yaml.values['model_building']['bins_pt']
    cuts_pre_filter = config_yaml.values['pre_filter_ml']['probability_min']

    def filter_trig(x):
        return sl.filter_df_prob(x, sl.build_cut_dict(pt_bins_trig, cuts_pre_filter))

    if is_dd:
        cols_assoc = cols_trig
        filter_assoc = None
        pt_bins_assoc = pt_bins_trig
    else:
        cols_assoc = config_yaml.values['pre_filter_ml']['cols_to_keep_assoc']
        # TODO: include electrons
        pt_bins_assoc = None
        filter_assoc = None

    files = [(reader.LazyFileLoader(x[0], ['RunNumber', 'EventNumber'], cols_trig),
              reader.LazyFileLoader(x[1]), ['RunNumber', 'EventNumber'], cols_assoc)
             for x in file_list]

    pairs = build_pairs_from_lazy(files, ('_t', '_a'), pt_bins_trig, pt_bins_assoc, filter_trig, filter_assoc, 1, is_dd)
    pairs.to_parquet(file_to_save)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("file_list", help='csv file with the locations of trigger and associated files')
    parser.add_argument("id", help='Number of the process, used to saved the produced files')
    parser.add_argument('-c' "--config", default=None, help='Yaml file with the configuration.')
    parser.add_argument('-d' "--process_dd", default=None, help='Yaml file with the configuration.')

    args = parser.parse_args()
