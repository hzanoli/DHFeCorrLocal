class Model:

    def __init__(self, dataset_name, yaml_config):

        if models_pt is None:
            models_pt = []
        if pt_bins is None:
            pt_bins = []
        if features is None:
            features = []

        self.pt_bins = pt_bins
        self.models_pt = models_pt
        self.features = []

    def load_models(self):

    def predict(self, data):
