from __future__ import print_function
import pandas as pd
import numpy as np

import yaml
import matplotlib.pyplot as plt
import warnings
from matplotlib.colors import LogNorm, Normalize


def feature_dict(df, name):
    """Create a dict of (QA feat) -> (dataframe feat) for a particle identified as 'name'. 
    Created to handle particle/antiparticle cases. 
    df should point to the dataframe where the QA takes place.
    name is the name of the particle ("D0", "D0bar", etc)
    """
    features = df.columns

    # names of the features with are particle dependent
    dependent_name = tuple([x[:-len(name)] for x in features if x.endswith(name)])

    # check all the ones that start with theses names
    dependent = [x for x in features if x.startswith(dependent_name)]

    # save only the ones with are relevant for this particle
    dependent_for_this_particle = [x + name for x in dependent_name]

    # remove all the dependent ones, even the ones from the antiparticles
    independent = [x for x in features if x not in dependent]

    ind_variables_dict = dict(zip(independent, independent))  # they have the same name in qa and in the DF
    dependent_variables_dict = dict(zip(dependent_name, dependent_for_this_particle))

    # creates a dict with the only the independents and dependents for this particle
    ind_variables_dict.update(dependent_variables_dict)
    return ind_variables_dict


class QAConfig(object):
    """Class to store the configuration used during the QA"""

    def __init__(self, selection_file="default_config_local.yaml", name="D0"):
        with open(selection_file, "r") as document:
            try:
                config = yaml.safe_load(document)
            except yaml.YAMLError as exc:
                print(exc)
            except:
                print("failed to load the file")
                raise

        self.file = config
        self.data = self.file["qa"][name]


def plot_density(df, bin_values, ax, title=None, range_y=(0, 1),
                 scale_x='linear', norm=Normalize,
                 **kwargs):
    """Plots a histogram as small density plot. Easy to check if cuts are applied"""

    # check if int or list is given for the bins
    if type(bin_values) is int:
        range_x = (min(df), max(df))
        bin_values = np.linspace(range_x[0], range_x[1], bin_values)
    else:
        range_x = (min(bin_values), max(bin_values))
    y_values = np.repeat(0.5, len(df))
    ax.hist2d(df, y_values, bins=(bin_values, range_y), range=(range_x, range_y), norm=norm())
    ax.set_xscale(scale_x, **kwargs)

    ax.yaxis.set_visible(False)
    ax.set_title(title)

    return bin_values


def plot_histogram(df, bin_values, ax, title=None, scale_x='linear', **kwargs):
    # noinspection SpellCheckingInspection
    """Plot the histogram for df (should be a array-like) with bins given by bin_values.
        The axes is mandatory. kwargs are passed to the set_xscale and set_xscale only."""

    # check if int or list is given for the bins
    if type(bin_values) is int:
        range_x = (min(df), max(df))
        bin_values = np.linspace(range_x[0], range_x[1], bin_values)

    ax.hist(df, bin_values, density=False)
    ax.set_xscale(scale_x, **kwargs)
    ax.set_yscale(scale_x, **kwargs)
    ax.set_title(title)

    return bin_values


d_sel_stat = dict({"D0": 1, "D0bar": 2, "both": 3})

log_y_plots = ['Pt']

qa_dir = "qa/"

tex_exec = '/Library/TeX/texbin/pdflatex'
config_name = "tree_v1"

qa_dir += config_name + "_"


def hist_with_errors(variable, bins_n='auto', ax=0, norm=False, return_values=False):
    """Plot a histogram with uncertainties. The errorbars are calculated using sqrt(number of entries).
    if ax is not provided, if will be created with the default settings from matplotlib."""
    try:
        counts, bin_edges = np.histogram(variable, bins=bins_n)
    except:
        warnings.warn("Automatic bin_edges failed for variable " + str(variable) + ". Replacing with 100",
                      RuntimeWarning)
        counts, bin_edges = np.histogram(variable, bins=100)

    counts = np.array(counts)
    bin_edges = np.array(bin_edges)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    y_error = np.sqrt(counts)
    x_err = 0.5 * (bin_edges[1:] - bin_edges[:-1])

    if norm:
        total = counts.sum()
        counts = counts / total
        y_error = counts / total

    if ax == 0:
        fig, ax = plt.subplots()

    ax.errorbar(bin_centers, counts, yerr=y_error, xerr=x_err, fmt='o')

    if return_values:
        return ax, counts, y_error, bin_edges

    return ax, bin_edges


def plot_cuts_part_antipart(before, after, cuts, name_qa, suffix='', plot_hist=False):
    """Wraper function used to:
    -Build the dictionaries for particles and antiparticles
    -Separate particles and antiparticles
    - Call plotter function for each of them.
    This should be the main entry point for D mesons."""
    particle_name = cuts.particle_name
    col_dict_part = feature_dict(before, particle_name + suffix)
    col_dict_anti = feature_dict(before, particle_name + 'bar' + suffix)

    after_part = after[after['IsSelected' + particle_name + suffix]]
    after_anti = after[after['IsSelected' + particle_name + 'bar' + suffix]]

    plots_part = plot_cuts(before, after_part, name_qa, col_dict_part, cuts=cuts, plot_hist=plot_hist)
    plots_anti = plot_cuts(before, after_anti, name_qa, col_dict_anti, cuts=cuts, plot_hist=plot_hist)

    return plots_part, plots_anti


def plot_cuts(df_1, df_2, names_qa, col_names_map=None, cuts=None, plot_hist=False):
    """Compare the features in df_1 (left) and df_2 (right). It can be used directly if the no cuts depend if 
    the candidate is a particle/antiparticle, such as the electrons"""

    variables_to_remove = ['GridPID', 'EventNumber', 'PtBin']

    col = col_names_map
    if col_names_map is None:
        warnings.warn("No map of qa names to DataFrame provided. Assuming they are the same. ", RuntimeWarning)
        col = dict(zip(df_1.columns, df_1.columns))

    for var in variables_to_remove:
        try:
            col.pop(var)
        except KeyError:
            pass

    pt_bin = df_1['PtBin'].iloc[0]

    try:
        selection_pt = cuts.selection_for_ptbin(pt_bin)
    except:
        pass

    size_v = 2.0

    if plot_hist:
        size_v = 5.0

    plots = list()
    for col_v, i in zip(col.keys(), range(len(col.keys()))):
        fig, ax = plt.subplots(1, 2, figsize=(16, size_v), constrained_layout=True)

        try:
            scale_x = names_qa[col_v]['scale_x']
        except:
            scale_x = 'linear'
        try:
            name = names_qa[col_v]['name']
        except:
            name = col

        normalization = Normalize

        try:
            scale_y = names_qa[col_v]['scale_y']
            if (scale_y == 'symlog') or (scale_y == 'log'):
                normalization = LogNorm
        except:
            pass

        kwargs = dict()

        try:
            linthreshy = names_qa[col_v]['linthreshx']
            kwargs.update({'linthreshx': linthreshy})
        except KeyError:
            pass
        # axes to hold the title

        fig.suptitle(name)

        if (plot_hist):
            bins = plot_histogram(df_1[col[col_v]], names_qa[col_v]['bins'], ax[0],
                                  None, scale_x=scale_x, **kwargs)
            plot_histogram(df_2[col[col_v]], bins, ax[1], None, scale_x=scale_x, **kwargs)

        else:
            bins = plot_density(df_1[col[col_v]], names_qa[col_v]['bins'], ax[0],
                                None, scale_x=scale_x, norm=normalization, **kwargs)
            plot_density(df_2[col[col_v]], bins, ax[1], None, scale_x=scale_x, norm=normalization, **kwargs)

        try:
            range_y = ax[1].get_ylim()
            range_x = ax[1].get_xlim()
            y_max = max(ax[1].get_ylim())

            if float(selection_pt.loc[col_v]) < min(range_x):
                ax[1].vlines(min(range_x) + 1.05(range_x[1] - range_x[0]), min(range_y), 1.05 * y_max,
                             colors=['blue'], linestyles='solid')
            else:
                ax[1].vlines(float(selection_pt.loc[col_v]), min(range_y), 1.05 * y_max,
                             colors=['blue'], linestyles='solid')
            position = 'right'
            ax[1].set_ylim(range_y)
            try:
                if cuts.is_max_selection(col_v):
                    position = "left"

                if float(selection_pt.loc[col_v]) < min(range_x):
                    warnings.warn('The minimum bin value for col: ' + str(col_v) + (
                        ' is less than the x value.It will not be shown'), RuntimeWarning)
                else:
                    text = ax[1].text(selection_pt.loc[col_v], 0.5, '  ' + str(selection_pt.loc[col_v]) + ' ',
                                      color="black", ha=position, va="center", fontweight="light")
                # text.set_clip_on(False)
            except:
                raise ValueError("not possible to create text")
        except:
            pass
        plots.append(fig)

    return plots
