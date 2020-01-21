def submit_make_pairs(dataset_name, config, trigger, assoc):
    pass


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("dataset_name", help='Name of the dataset')
    parser.add_argument('-c' "--config", default=None, help='Yaml file with the configuration.')
    parser.add_argument('-t' "--trigger", dest='dmeson', help='Name of the trigger particle')
    parser.add_argument('-a' "--associated", default='dmeson', help='Name of the associated particle')

    args = parser.parse_args()
