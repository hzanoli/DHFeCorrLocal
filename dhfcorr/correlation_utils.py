import pandas as pd
import numpy as np


def convert_to_range(dphi):
    if dphi > 3. * np.pi / 2.:
        dphi = dphi - 2. * np.pi
    if dphi < -np.pi / 2.:
        dphi = dphi + 2. * np.pi

    if dphi < 0.:
        dphi = -dphi
    if dphi > np.pi:
        dphi = 2 * np.pi - dphi

    return dphi


def reduce_to_single_particle(correlation, suffix):
    particle = correlation.groupby(by=[correlation.index, correlation['Id' + suffix]],
                                   sort=False, as_index=False).nth(0)
    cols_to_keep = [x for x in correlation.columns if x.endswith('Bin')]
    cols_to_keep += [x for x in correlation.columns if x.endswith(suffix)]
    particle = particle[cols_to_keep]

    return particle


def compute_angular_differences(correlation, suffixes, bins_phi, bins_eta, **kwargs):
    correlation['DeltaPhi'] = (correlation['Phi' + suffixes[0]] - correlation['Phi' + suffixes[1]]).apply(
        convert_to_range)
    correlation['DeltaEta'] = (correlation['Eta' + suffixes[0]] - correlation['Eta' + suffixes[1]])

    # Calculate the bins for angular quantities
    correlation['DeltaPhiBin'] = pd.cut(correlation['DeltaPhi'], bins_phi)
    correlation['DeltaEtaBin'] = pd.cut(correlation['DeltaEta'], bins_eta)

    # Calculate the weight of the pair
    correlation['Weight'] = correlation['Weight' + suffixes[0]] * correlation['Weight' + suffixes[1]]
    # Save the weight square that will be useful to calculate the errors
    correlation['WeightSquare'] = correlation['Weight'] ** 2
