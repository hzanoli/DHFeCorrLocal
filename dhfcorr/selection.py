from __future__ import print_function

import numpy as np
import pandas as pd
import warnings


class Cuts(object):

    def feature_names(self):
        """Get the features names that have a selection cut defined"""
        return list(self.cut_df.columns)

    def selection_for_ptbin(self, pt_bin):
        """Get the cut selection for a specific bin"""
        return pd.Series(self.cut_df.iloc[int(pt_bin)])

    def is_min_selection(self, feat):
        return feat in self.min_features

    def is_max_selection(self, feat):
        return feat in self.max_features

    def n_ptbins(self):
        return int(len(self.pt_bins) - 1)

    def __init__(self, yaml_config, particle='D0'):
        """Default constructor. yaml_file should come from the class CutsYaml. The particle is set as Default to D0 """
        try:
            d_meson_cuts = yaml_config.values[particle]['cuts']
        except KeyError as key_error:
            print("The particle " + str(particle) + " cuts were not found. The error raised is:")
            print(key_error)

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
        self.part_dep_cuts = tuple(yaml_config.values[particle]['particle_dependent_variables'])
        self.particle_mass = float(yaml_config.values[particle]['particle_mass'])
        self.particle_name = str(yaml_config.values[particle]['particle_name'])
        self.features_absolute = tuple(yaml_config.values[particle]['features_abs'])


def feature_dict(cuts, name):
    """Create a dict of (cut feat) -> (dataframe feat) for a particle identified as 'name' """
    particle_dependent_variables = cuts.part_dep_cuts

    independent_variables = [x for x in cuts.feature_names() if x not in particle_dependent_variables]
    ind_variables_dict = dict(zip(independent_variables, independent_variables))

    dependent_variables_names = [x + name for x in particle_dependent_variables]
    dependent_variables_dict = dict(zip(particle_dependent_variables, dependent_variables_names))

    ind_variables_dict.update(dependent_variables_dict)

    return ind_variables_dict


def apply_cuts_pt(df, cuts, col_dict, pt_bin=None, select_in_pt_bins=True):
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
        selected_condition = df[col_dict[feat]] >= float(selection_pt.loc[feat])
        filtered = filtered & selected_condition

    for feat in cuts.max_features:
        selected_condition = df[col_dict[feat]] <= float(selection_pt.loc[feat])
        filtered = filtered & selected_condition

    for feat in cuts.bool_features:
        if bool(selection_pt.loc[feat]):  # Use only if True, if False ignore (does not apply the cut)
            selected_condition = df[col_dict[feat]] == bool(selection_pt.loc[feat])
            filtered = filtered & selected_condition

    return filtered


def apply_part_antipart_selection(df, cuts):
    """Checks if the particles/antiparticles in df fulfill the selection described in 'cuts'
    Assumes that the bins were already computed with pd.cut and that the df.name points to the Pt Bin 
    
    Returns a df with only with the selected particles. 
    The result adds two new columns with the selection results for the particle and antiparticles
    """
    particle_name = cuts.particle_name
    col_dict_part = feature_dict(cuts, particle_name)
    col_dict_anti = feature_dict(cuts, particle_name + "bar")

    filtered_part = apply_cuts_pt(df, cuts, col_dict_part)
    filtered_anti = apply_cuts_pt(df, cuts, col_dict_anti)

    df['IsSelected' + particle_name] = filtered_part
    df['IsSelected' + particle_name + "bar"] = filtered_anti

    filtered = filtered_part | filtered_anti

    return df[filtered]


def filter_in_pt_bins(df, cuts, add_pt_bin_feat=False):
    """General warper to perform the selection of particles in df described in cuts.
    The bins are defines as [a,b), so they include the lowest value but not the highest value."""

    # cut the dataframe in pt bins. The bin
    pt_bins = pd.cut(df['Pt'], cuts.pt_bins, labels=False, include_lowest=True)
    pass_cuts = df.groupby(by=pt_bins).apply(lambda x: apply_part_antipart_selection(x, cuts))

    if add_pt_bin_feat:
        df['PtBin'] = pt_bins
        df.set_index([df['PtBin'], df.index], inplace=True)
        df.sort_index(inplace=True)
        pass_cuts.index = pass_cuts.index.rename(['PtBin', ''])

    return pass_cuts


def build_add_features_dmeson(df, cuts, dmeson_type=None):
    """Build features which will be used during the selection for D mesons. Any additional features can be added here.
    d_meson_type is not implemented, but it will be used to provide interface to different d mesons"""
    particle_mass = cuts.particle_mass
    particle_name = cuts.particle_name

    # df['D0Prod'] = df['D0Daughter1'] * df['D0Daughter0']
    df['DeltaM'] = df['InvMass'] - particle_mass

    # df['DeltaM' + particle_name] = df['InvMass' + particle_name] - particle_mass
    # df['DeltaM' + particle_name + 'bar'] = df['InvMass' + particle_name + 'bar'] - particle_mass

    # df['PID' + particle_name] = (df['SelectionStatus'] == 1) | (df['SelectionStatus'] == 3)
    # df['PID' + particle_name + 'bar'] = (df['SelectionStatus'] == 2) | (df['SelectionStatus'] == 3)

    for col in cuts.features_absolute:
        if col in cuts.part_dep_cuts:
            df[col + particle_name] = np.abs(df[col + particle_name])
            df[col + particle_name + 'bar'] = np.abs(df[col + particle_name + 'bar'])
        else:
            df[col] = np.abs(df[col])


def select_inv_mass(df, part_name):
    try:
        particle_cand = df['InvMass' + part_name][df['IsSeleced' + part_name]]
        antipart_cand = df['InvMass' + part_name + "bar"][df['IsSeleced' + part_name + 'bar']]
    except KeyError:
        return None
    mass_values = pd.concat([particle_cand, antipart_cand])
    return mass_values


def build_add_features_electron(df, cuts, e_type=None):
    pass
