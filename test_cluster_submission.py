import numpy as np
import h2o

bins = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10., 12., 16., 9999.])

h2o.init()

dataset = h2o.import_file('..data/test.parquet')
dataset['PtBin'] = dataset['Pt'].cut(list(bins))

model = h2o.load_model('h2o_pt_0')

prediction = model.predict(dataset)
dataset['prediction'] = prediction['p1']

df = dataset[dataset['prediction'] > 0.9].as_data_frame()

df.to_parquet('test_result.parquet')

h2o.cluster().shutdown()
