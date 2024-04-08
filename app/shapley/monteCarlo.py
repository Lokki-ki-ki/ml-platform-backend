import numpy as np
from copy import deepcopy
from dotenv import load_dotenv, find_dotenv
from app.utils.ipfs_utils import download_file_from_ipfs, formatting_weights
from app.utils.ml_utils import initialize_sample_model, evaluate_model_by_weight, calculate_avg_weights
from pathlib import Path
import os

_ = load_dotenv(find_dotenv())
FOLDER = os.getenv("TEMP_FOLDER")
TEMP_FILE_PATH = Path(__file__).parent.parent / FOLDER

# ********** Common Function **********
def download_clients_weights(client_weights):
    """
    Convert client_id: weights_ipfs_address to client_id: weights_local_path
    """
    clientsToWeights = {}
    for client_id, weights_address in client_weights.items():
        path = download_file_from_ipfs(weights_address, client_id)
        clientsToWeights[path[0]] = path[1]
    return clientsToWeights
# **************************************

def truncated_monte_carlo_shapley(sample_model, sample_weights, test_data, test_labels, evaluation_func, clientsToWeights, performance_tolerance=0.0002, max_iter=1):
    # sample_weights = sample_model.get_weights()
    n = len(clientsToWeights)
    total_tup = tuple(clientsToWeights.keys())
    print(total_tup)
    agg_D_weights = calculate_avg_weights(clientsToWeights, total_tup, sample_model)
    # make a copy of the model instead of reference
    agg_D = sample_model
    agg_D.set_weights(agg_D_weights)
    agg_D.save_weights(TEMP_FILE_PATH / "temp_weights.weights.h5")
    _, v_D = evaluation_func(TEMP_FILE_PATH / "temp_weights.weights.h5", 0, sample_model, test_data, test_labels)[1]
    print(v_D, 'v_D')
    marginal_contributions = { i: 0 for i in clientsToWeights.keys() }

    for t in range(1, max_iter + 1):
        permutation = np.random.permutation(n)
        permutation = tuple(map(lambda x: total_tup[x], permutation))
        j = 0
        model = sample_model
        model.set_weights(sample_weights)
        model.save_weights(TEMP_FILE_PATH / "temp_weights.weights.h5")
        _, v_previous = evaluation_func(TEMP_FILE_PATH / "temp_weights.weights.h5", j, sample_model, test_data, test_labels)[1]
        print(v_previous, 'v_previous')
        # v_previous = 0
        for j in range(1, n + 1):
            if abs(v_D - v_previous) < performance_tolerance:
                v_current = v_previous
            else:
                agg_model = sample_model
                tup = permutation[:j]
                agg_weights = calculate_avg_weights(clientsToWeights, tup, sample_model)
                agg_model.set_weights(agg_weights)
                agg_model.save_weights(TEMP_FILE_PATH / "temp_weights.weights.h5")
                _, v_current = evaluation_func(TEMP_FILE_PATH / "temp_weights.weights.h5", j, sample_model, test_data, test_labels)[1]
            print(v_current, 'v_current')
            client = permutation[j-1]
            marginal_contributions[client] = (t-1)/t * marginal_contributions[client] + (v_current - v_previous) / t
    return marginal_contributions


import h5py
if __name__ == '__main__':
    clientsToSubmissions = {"1": "QmdfzatZMtMmaWMTNsJuvCQcBbHAQFSGTrYLi6CiZ5fWTi", "2": "QmX4T5dLBrubvGmUJnkM1tP1j8oPC5ZwDMUWdDR1R9wWNN",
                            "3": "QmTCxkXPG9fetQs1mP6QmQDWAx5vUc325u8AndPquJQv62", "4": "QmPZeG6uJsX1EsKd4BeR5cPNBScAwpU7CLXRGXrEgbUpSV", "5": "Qmc8VCb4DWZbXxhqcCCsqdVoYiGqrmkVbQ4fHPhYopRwS1"}
    clientsToWeights = formatting_weights(download_clients_weights(clientsToSubmissions))
    test_data_add = "Qmeo6yw83vLYX3zkgPDiGK24FB99pz1PX1EngLHLPyko76"
    test_labels_add = "QmWTBKyjTsuHHq6kHdCnbvF3PQZDQa2pincoyAc8DzisFX"
    test_data_path = download_file_from_ipfs(test_data_add, 0)[1]
    test_labels_path = download_file_from_ipfs(test_labels_add, 0)[1]
    with h5py.File(test_data_path, "r") as hf:
        data = hf["test_data"][:]
        hf.close()
    with h5py.File(test_labels_path, "r") as hf:
        labels = hf["test_labels"][:]
        hf.close()
    test_data = data
    test_labels = labels
    sample_model = initialize_sample_model()

    res = truncated_monte_carlo_shapley(sample_model, test_data, test_labels, evaluate_model_by_weight, clientsToWeights)
    print(res)



