import pandas as pd
from qwak_mock import real_time_client


def test_churn_classifier(real_time_client):

    feature_vector = [{"table_name":"churn_to_pred_1"}]

    predict_res = real_time_client.predict(feature_vector)
    assert predict_res['size'].values[0] == 1