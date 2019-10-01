import h2o
import time
import argparse

from dhfcorr.ml.setup_sge_h2o import setup_h2o_cluster_sge


def train_model():
    train_path = "/data2/data/d_hfe/processing/D0_HMV0/ml-dataset/ml_sample_0.parquet"
    dataset = h2o.import_file(train_path)
    target = 'CandidateType'
    dataset[target] = dataset[target].asfactor()

    train, test = dataset.split_frame([0.5], seed=1234)
    param = {"ntrees": 50,
             "max_depth": 10,
             "learn_rate": 0.02,
             "sample_rate": 0.7,
             "col_sample_rate_per_tree": 0.9,
             "min_rows": 5,
             "seed": 4241,
             "score_tree_interval": 100,
             }
    from h2o.estimators import H2OXGBoostEstimator
    model = H2OXGBoostEstimator(**param)
    start = time.time()
    model.train(y=target, training_frame=train, validation_frame=test)
    print(model)
    training_time = time.time() - start

    return training_time


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("cloud_name", help="Name of the cloud use in H2O")
    args = parser.parse_args()

    setup_h2o_cluster_sge(args.cloud_name)
    connect = h2o.connect()

    time_spent = train_model()
    print("Time spent to train:")
    print(time_spent)

    h2o.cluster().shutdown()
