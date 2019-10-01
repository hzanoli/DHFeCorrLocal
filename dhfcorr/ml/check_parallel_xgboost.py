import h2o
import time


def train_model(n_cores):
    h2o.init(nthreads=n_cores, max_mem_size='16G')

    train_path = 'higgs_train_imbalance_100k.csv'
    test_path = 'higgs_test_imbalance_100k.csv'
    df_train = h2o.import_file(train_path)
    df_test = h2o.import_file(test_path)
    df_valid = df_test
    # Transform first feature into categorical feature
    df_train[0] = df_train[0].asfactor()
    df_valid[0] = df_valid[0].asfactor()

    param = {"ntrees": 100,
             "max_depth": 10,
             "learn_rate": 0.02,
             "sample_rate": 0.7,
             "col_sample_rate_per_tree": 0.9,
             "min_rows": 5,
             "seed": 4241,
             "score_tree_interval": 100
             }
    from h2o.estimators import H2OXGBoostEstimator
    model = H2OXGBoostEstimator(**param)
    start = time.time()
    model.train(x=list(range(1, df_train.shape[1])), y=0, training_frame=df_train, validation_frame=df_valid)
    training_time = time.time() - start
    h2o.cluster().shutdown()
    time.sleep(3)

    return training_time


n_cores_values = [1, 4, 8]
total_time = [train_model(x) for x in n_cores_values]
print(total_time)
