import ROOT


class FitHF:
    """ Class used to fit results using the AliHFInvMassFitter and keep track of the results

    Attributes
    ----------
    fit_obj : ROOT.AliHFInvMassFitter

    """

    def fit_inv_mass_root(self, histogram, config_inv_mass, config_inv_mass_def):
        """"Fits the invariant mass distribution using AliHFInvMassFitter.

        Parameters
        ----------
        histogram : ROOT.TH1
            The histogram that will be fitted.
        config_inv_mass : dict
            Values used to configure the AliHFInvMassFitter.
        config_inv_mass_def: dict
            Default values of config_inv_mass. In case of the the parameters in config_inv_mass is not available, it will be
             picked from it.

        Returns
        -------
        fit_mass : ROOT.AliHFInvMassFitter
            The fit mass object for this histogram

        Raises
        ------
        KeyError
            If the keywords (range, ) used to configure the AliHFInvMassFitter are not found on config_inv_mass.
        ValueError
            In case the one of the configurations in config_inv_mass (or config_inv_mass_def) is not consistent.

        """

        # Copy dict to avoid changes
        local_dict = config_inv_mass_def.copy()
        local_dict.update(config_inv_mass)

        try:
            minimum = local_dict['range'][0]
            maximum = local_dict['range'][1]
            if minimum > maximum:
                raise ValueError('Minimum invariant mass is higher than maximum invariant mass. Check limits.')
            try:
                bkg_func = getattr(ROOT.AliHFInvMassFitter, local_dict['bkg_func'])
            except AttributeError as err:
                print(err)
                raise ValueError("Value of background function not found on AliHFInvMassFitter.")
            try:
                sig_func = getattr(ROOT.AliHFInvMassFitter, local_dict['sig_func'])
            except AttributeError as err:
                print(err)
                raise ValueError("Value of background function not found on AliHFInvMassFitter.")

        except KeyError as kerr:
            print(kerr)
            print("Problem accessing the configuration of the mass fitter")
            raise

        fit_mass = ROOT.AliHFInvMassFitter(histogram, minimum, maximum, bkg_func, sig_func)
        fit_mass.MassFitter(False)

        return fit_mass

    def __init__(self, histogram, config_inv_mass, config_inv_mass_def):
        self.fit_inv_mass_root(histogram, config_inv_mass, config_inv_mass_def)
