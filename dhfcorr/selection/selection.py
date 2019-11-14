import numpy as np
import pandas as pd
import warnings
import dhfcorr.config_yaml as configyaml
import dhfcorr.definitions as definitions
import dhfcorr.io.data_reader as reader


class Cuts(object):

    def feature_names(self):
        """Get the features names that have a selection cut defined"""
        return list(self.cut_df.columns)

    def selection_for_ptbin(self, pt_bin):
        """Get the cut selection for a specific bin"""
        return self.cut_df.loc[pt_bin]

    def is_min_selection(self, feat):
        return feat in self.min_features

    def is_max_selection(self, feat):
        return feat in self.max_features

    def n_pt_bins(self):
        return int(len(self.pt_bins) - 1)

    def __init__(self, name_file, particle='D0'):
        """Default constructor. yaml_file should come from the class CutsYaml. The particle is set as Default to D0 """
        yaml_config = configyaml.ConfigYaml(name_file)
        try:
            d_meson_cuts = yaml_config.values[particle]['cuts']
        except KeyError as key_error:
            print(key_error)
            raise (ValueError, "The particle " + str(particle) + " cuts were not found.")

        # Save the cuts to a DataFrame
        self.cut_df = pd.DataFrame(d_meson_cuts).apply(pd.to_numeric, errors='ignore')
        self.cut_df.set_index('PtBin', inplace=True)

        # Change names to values with no -range, min_, max_
        names = [a.split('_')[0] for a in self.cut_df.columns]
        type_col = [a.split('_')[1] for a in self.cut_df.columns]  # save the type of cut

        self.range_features = [names[i] for i in range(len(names)) if type_col[i] == "range"]
        self.min_features = [names[i] for i in range(len(names)) if type_col[i] == "min"]
        self.max_features = [names[i] for i in range(len(names)) if type_col[i] == "max"]
        self.bool_features = [names[i] for i in range(len(names)) if type_col[i] == "bool"]

        self.cut_df.columns = names
        self.cut_type = type_col

        pt_ = self.cut_df['Pt']
        min_pt = [pt_[i][0] for i in range(len(pt_))]
        max_pt = [pt_[i][1] for i in range(len(pt_))]

        # Define basic selection variable types
        self.pt_bins = list(min_pt) + list([max_pt[-1]])

        # Change pt_bins to intervals
        mid_pt = (np.array(min_pt) + np.array(max_pt)) / 2.
        self.cut_df['PtBin'] = pd.cut(mid_pt, self.pt_bins)
        self.cut_df.set_index('PtBin', inplace=True)

        self.part_dep_cuts = tuple(yaml_config.values[particle]['particle_dependent_variables'])
        self.particle_mass = float(yaml_config.values[particle]['particle_mass'])
        self.particle_name = str(yaml_config.values[particle]['particle_name'])
        self.features_absolute = tuple(yaml_config.values[particle]['features_abs'])


def apply_cuts_pt(df, cuts, pt_bin=None, select_in_pt_bins=True):
    """Apply the selection cuts defined in 'cuts' to df.
    col_dict should containt the dict that maps the keys used in the selection class to the one in df.
    Range features are not yet implemented.
    Returns a list in True or False (the selection status)"""

    # TODO: implement range features
    if pt_bin is None:
        try:
            pt_bin = df.name
        except AttributeError:
            if select_in_pt_bins:
                pt_bin = 0
                warnings.warn('It is not possible to determine the pt bin. \
                    The value was set to 0. You can silence this warning by setting select_in_pt_bins to False.')
            else:
                pt_bin = 0
                pass

    selection_pt = cuts.selection_for_ptbin(pt_bin)
    filtered = (df[df.columns[0]] == df[df.columns[0]])  # always returns true, used to started all as true

    for feat in cuts.min_features:
        col = df[feat]
        if feat in cuts.features_absolute:
            col = np.abs(col)

        selected_condition = col >= float(selection_pt.loc[feat])
        filtered = filtered & selected_condition

    for feat in cuts.max_features:
        col = df[feat]
        if feat in cuts.features_absolute:
            col = np.abs(col)

        selected_condition = col <= float(selection_pt.loc[feat])
        filtered = filtered & selected_condition

    for feat in cuts.bool_features:
        if bool(selection_pt.loc[feat]):  # Use only if True, if False ignore (does not apply the cut)
            selected_condition = df[feat] == bool(selection_pt.loc[feat])
            filtered = filtered & selected_condition

    return df[filtered]


def filter_in_pt_bins(df, cuts):
    """General warper to perform the selection of particles in df described in cuts."""

    # cut the dataframe in pt bins.
    pt_bins = pd.cut(df['Pt'], cuts.pt_bins)
    cols_present = [x in df.columns for x in cuts.feature_names()]
    if not all(cols_present):
        raise ValueError('The following columns are specified in the cuts, but are not present in the DataFrame: \n'
                         '' + str([x for x in cuts.feature_names() if x not in df.columns]))

    pass_cuts = df.groupby(by=pt_bins).apply(lambda x: apply_cuts_pt(x, cuts))

    return pass_cuts


def build_additional_features_dmeson(df):
    """Build features which will be used during the selection for D mesons.
    Any additional features can be added here.
    """
    df['D0Prod'] = (df['D0Daughter1'] * df['D0Daughter0']).astype(np.float32)
    # Selected PID in the default PID selection
    particles = df['IsParticleCandidate']
    default_pid = df['SelectionStatusDefaultPID']
    df['PID'] = (default_pid == 3) | (particles & (default_pid == 1)) | (~particles & (default_pid == 2))


def get_true_dmesons(df):
    """"Used in simulations (MC). Select only the candidates which have the correct hypothesis at the reconstruction and
    generated level.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with the information. It must have the columns IsD (check if the particle is a D meson, particle
         or antiparticle, at generated level), IsParticleCandidate (reconstruction hypothesis) and IsParticle (generated
         level hypothesis).
    """
    particles = df['IsD'] & df['IsParticleCandidate'] & df['IsParticle']
    antiparticles = df['IsD'] & ~df['IsParticleCandidate'] & ~df['IsParticle']

    return df[particles | antiparticles]


def get_reflected_dmesons(df):
    """"Used in simulations (MC). Select only the candidates which have the incorrect hypothesis at the reconstruction
    and generated level.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with the information. It must have the columns IsD (check if the particle is a D meson, particle
         or antiparticle, at generated level), IsParticleCandidate (reconstruction hypothesis) and IsParticle (generated
         level hypothesis).
    """
    particles = df['IsD'] & df['IsParticleCandidate'] & df['IsParticle']
    antiparticles = df['IsD'] & ~df['IsParticleCandidate'] & ~df['IsParticle']

    return df[~(particles | antiparticles)]


def filter_df(df_x, cuts, suffix):
    pt_bin = df_x.name
    cut = float(cuts.loc[pt_bin])
    return df_x[df_x['prediction' + suffix] >= cut]


def build_additional_features_electron(df):
    pass
