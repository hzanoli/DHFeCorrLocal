import yaml
from dhfcorr.definitions import ROOT_DIR


class ConfigYaml(object):
    """Hold the yaml file containing the configuration"""

    def __init__(self, selection_file=None, default_file=ROOT_DIR + "/config/default_config_local.yaml"):
        """Default constructor. selection_file is the yaml file with the configuration"""

        # Default configuration
        default_config = ''

        with open(default_file, "r") as document:
            try:
                default_config = yaml.safe_load(document)
            except yaml.YAMLError as exc:
                print(exc)
                raise FileNotFoundError('Default file not found or with problems. Check error above.')
        if default_config == '':
            raise FileNotFoundError('Empty default configuration. Check the YAML file.')

        self.values = default_config

        config = ''

        # https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
        def update(d, u):
            import collections
            for k, v in u.items():
                if isinstance(v, collections.Mapping):
                    d[k] = update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d

        if selection_file is not None:
            print(selection_file)
            with open(selection_file, "r") as document:
                try:
                    config = yaml.safe_load(document)
                except yaml.YAMLError as exc:
                    print(exc)
                    raise FileNotFoundError('User configuration file not found or with problems. Check error above.')
            if config == '':
                raise FileNotFoundError('Empty user configuration. Check the YAML file.')
            else:
                update(self.values, config)
