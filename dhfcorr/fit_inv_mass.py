import pandas as pd
import ROOT
import dhfcorr.definitions as definitions
from dhfcorr.fit.fit1D import fit_d_meson_inv_mass

ROOT.TH1.AddDirectory(False)


def select_inv_mass(df, min_mass, max_mass, suffix=''):
    """"Select the rows of df dataframe which is its mass is between [min_mass, max_mass] and return a view.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with the D mesons. Should contain the column InvMass
    min_mass: float
        Minimum value that will be selected in the mass. Do not give it in number of sigmas.
    max_mass: float
        Maximum value that will be selected in the mass. Do not give it in number of sigmas.
    suffix: str
        Suffix used to modify the name of the DataFrame (e.g. '_d')
    Returns
    -------
    df_selected: pd.DataFrame
        Dataframe of the selected particles. Returns a view.

    """
    selected = ((df['InvMass' + suffix] >= min_mass) & (df['InvMass' + suffix] <= max_mass))  # Check max_mass
    return df[selected]


if __name__ == '__main__':
    """"Fit the invariant mass of the pairs.
    The first argument is the name of the file that contain the pairs of D-e. If none is provided, the default name 
    from definitions.py is used.
    The second argument is the name of the configuration yaml file. If none is provided, the default one is used.
    """

    import sys

    try:
        file_name = str(sys.argv[1])
    except IndexError:
        file_name = definitions.pairs

    try:
        config_file = sys.argv[2]
    except IndexError:
        config_file = None

    fit_d_meson_inv_mass(config_file)
