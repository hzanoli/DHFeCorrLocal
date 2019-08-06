from joblib import load
import numpy as np
import warnings
import sklearn
import pandas as pd
import dhfcorr.selection as sl


def select_using_ml(df, file_name, cuts):
    model = load(file_name)

    part = df[variables_part]
    anti_part = df[variables_anti]

    df['IsSelected' + particle_name] = model.decision_function(part)
    df['IsSelected' + particle_name + "bar"] = model.decision_function(anti_part)

    return df


def decision_function(df, base_file_name='bdt_'):
    pt = int(df.name)
    particle_name = 'D0'

    file_name = base_file_name + str(pt) + '.joblib'
    model = load(file_name)

    variables_part = ['NormDecayLengthXY', 'CosP', 'CosPXY', 'ImpParXY', 'DCA', 'Normd0MeasMinusExp',
                      'PtDaughter0', 'PtDaughter1', 'ReducedChi2', 'D0Prod', 'CosTsD0', 'PIDD0',
                      'D0Daughter0', 'D0Daughter1']

    variables_anti = ['NormDecayLengthXY', 'CosP', 'CosPXY', 'ImpParXY', 'DCA', 'Normd0MeasMinusExp',
                      'PtDaughter1', 'PtDaughter0', 'ReducedChi2', 'D0Prod', 'CosTsD0bar', 'PIDD0bar',
                      'D0Daughter1', 'D0Daughter0']

    part = df[variables_part]
    anti_part = df[variables_anti]

    df['IsSelected' + particle_name] = model.decision_function(part)
    df['IsSelected' + particle_name + "bar"] = model.decision_function(anti_part)

    return df





def col_names_for_particles(df, particle_name, particle_dependent_variables):
    new_cols = list()
    for col in df.columns:
        col_name = col
        for feat in particle_dependent_variables:
            if col == (feat + particle_name):
                col_name = feat
        new_cols.append(col_name)
    return new_cols


def invert_daughters(df, filter=False):
    if filter:
        df['temp'] = df['PtDaughter0']
        df.loc[np.invert(df['IsParticle']), 'PtDaughter0'] = df.loc[np.invert(df['IsParticle']), 'PtDaughter1']
        df.loc[np.invert(df['IsParticle']), 'PtDaughter1'] = df.loc[np.invert(df['IsParticle']), 'temp']

        df['temp'] = df['D0Daughter0']
        df.loc[np.invert(df['IsParticle']), 'D0Daughter0'] = df.loc[np.invert(df['IsParticle']), 'D0Daughter1']
        df.loc[np.invert(df['IsParticle']), 'D0Daughter1'] = df.loc[np.invert(df['IsParticle']), 'temp']
        df.drop('temp', axis='columns', inplace=True)

    else:
        df['temp'] = df['PtDaughter0']
        df['PtDaughter0'] = df['PtDaughter1']
        df['PtDaughter1'] = df['temp']

        df['temp'] = df['D0Daughter0']
        df['D0Daughter0'] = df['D0Daughter1']
        df['D0Daughter1'] = df['temp']

        df.drop('temp', axis='columns', inplace=True)


def duplicate_candidates(df, d_cuts):
    if df is None:
        return None

    df_part = df.copy()
    df_part['ID'] = df_part.index
    # build additional features
    particle_name = d_cuts.particle_name
    df_antipart = df_part.copy()

    # duplicates the content, but uses the correct particle/antiparticle variables
    df_part.columns = col_names_for_particles(df_part, particle_name, d_cuts.part_dep_cuts)
    df_antipart.columns = col_names_for_particles(df_antipart, particle_name + 'bar', d_cuts.part_dep_cuts)

    df_part['IsParticle'] = True
    df_antipart['IsParticle'] = False

    invert_daughters(df_antipart, filter=False)
    df = pd.concat([df_part, df_antipart], sort=True)

    # Divide in Pt Bins
    # df['PtBin'] = pd.cut(df['Pt'], bins=list(bins), include_lowest=True)

    cols_to_delete_particle = [col + d_cuts.particle_name for col in d_cuts.part_dep_cuts]
    cols_to_delete_anti = [col + d_cuts.particle_name + 'bar' for col in d_cuts.part_dep_cuts]
    cols_to_delete = cols_to_delete_particle + cols_to_delete_anti

    df.drop(cols_to_delete, axis='columns', inplace=True)

    # Drop particles with any missing value: they should come from the 'PtBin'

    return df
