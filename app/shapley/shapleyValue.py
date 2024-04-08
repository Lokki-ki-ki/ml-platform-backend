from itertools import combinations
import numpy as np
from pathlib import Path
import math
import os
import h5py
from dotenv import find_dotenv, load_dotenv
from app.utils.ipfs_utils import download_file_from_ipfs, formatting_weights
from app.utils.ml_utils import initialize_sample_model, evaluate_model_by_weight

_ = load_dotenv(find_dotenv())
FOLDER = os.getenv("TEMP_FOLDER")
TEMP_FILE_PATH = Path(__file__).parent.parent / FOLDER

# ********** Common Function **********

def logistic_function(self, x):
    return 1 / (1 + math.exp(-x))
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


def get_all_combinations(clients_ids):
    """
    Input: list of clients' ids such as [1, 2, 3] or a dictionary
    For each possible permutation length, generate permutations
    Returns [[1, 2, 3], [2, 1], [3, 1]]
    """
    if (type(clients_ids) == dict):
        clients_ids = list(clients_ids.keys())
    
    all_combinations = []
    for i in range(1, len(clients_ids)+1):
        all_combinations.extend(list(combinations(clients_ids, i)))
    return all_combinations


def sort_tuple(tup):
    """
    Input: a tuple, such as (1,2,3)
    Output: a sorted string
    """
    result = list(tup)
    result.sort()
    str_result = ''.join(list(map(lambda x: str(x), result)))
    return str_result

def calculate_avg_weights(clientToWeights, tup, sample_model):
    weights = []
    for i in tup:
        model = sample_model
        model.load_weights(clientToWeights[i])
        weights.append(model.get_weights())
    avg_weights = []
    for i in range(len(weights[0])):
        avg_weights.append(np.average([model[i] for model in weights], axis=0))
    return avg_weights

def calculate_accuracy_for_permutations(clientToWeights, sample_model, sample_weights, evaluation_function, test_data, test_labels):
    """
    Input: a dictionary of permutations and a dictionary of weights
    Output: a dictionary of permutations and their corresponding accuracy
    """
    dict_of_acc = {}
    permutations = get_all_combinations(clientToWeights)
    for tup in permutations:
        key = sort_tuple(tup)
        avg_weights = calculate_avg_weights(clientToWeights, tup, sample_model)
        model = sample_model
        model.set_weights(avg_weights)
        model.save_weights(TEMP_FILE_PATH / "temp_weights.h5")
        acc = evaluation_function(TEMP_FILE_PATH / "temp_weights.h5", key, sample_model, test_data, test_labels)
        dict_of_acc[key] = acc[1][1]
    model.set_weights(sample_weights)
    dict_of_acc[''] = model.evaluate(test_data, test_labels)[1]
    return dict_of_acc

def calculate_weight(N, S):
    """
    Input: number of clients, number of clients in a subset
    Output: weight
    """
    return math.factorial(N - S) * math.factorial(S - 1) / math.factorial(N)

def prepare_SV(clientsToWeights, sample_model, sample_weights, evaluate_model_by_weight, test_data, test_labels):
    """
    Input: a dictionary of clients' weights, a sample model, a function to evaluate the model, test data and labels
    Output: a dictionary of permutations and their corresponding accuracy
    """
    clientsToWeights = formatting_weights(clientsToWeights)
    permutation = get_all_combinations(clientsToWeights)
    dict_of_acc = calculate_accuracy_for_permutations(clientsToWeights, sample_model, sample_weights, evaluate_model_by_weight, test_data, test_labels)
    return permutation, dict_of_acc

def calculate_SV(permutations, dict_of_acc, client_id, N,if_logistic=True):
    """
    permutations: list of tuples, each tuple is a permutation of clients, such as [(1,2,3), (2,1,3), (3,2,1)]
    client_id: the client we want to calculate the SV for, such as 1
    N: total number of clients
    """
    print("Calculating SV for client {}".format(client_id))
    SV = 0
    for tup in permutations:
        if client_id in tup:
            S = len(tup)
            tag = sort_tuple(tup)
            current_acc = dict_of_acc[tag]
            minusset = set(tup) - set([client_id])
            minusset_tag = sort_tuple(minusset)
            minusset_acc = dict_of_acc[minusset_tag]
            weight = calculate_weight(N, S)
            SV += weight * (current_acc - minusset_acc)
    print("SV for client {} is {}".format(client_id, SV))
    return SV


if __name__ == '__main__':
    clientsToSubmissions = {"1": "QmdfzatZMtMmaWMTNsJuvCQcBbHAQFSGTrYLi6CiZ5fWTi", "2": "QmX4T5dLBrubvGmUJnkM1tP1j8oPC5ZwDMUWdDR1R9wWNN",
                            "3": "QmTCxkXPG9fetQs1mP6QmQDWAx5vUc325u8AndPquJQv62", "4": "QmPZeG6uJsX1EsKd4BeR5cPNBScAwpU7CLXRGXrEgbUpSV", "5": "Qmc8VCb4DWZbXxhqcCCsqdVoYiGqrmkVbQ4fHPhYopRwS1"}
    clientsToWeights = download_clients_weights(clientsToSubmissions)
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

    permutations, dict_of_acc = prepare_SV(clientsToWeights, sample_model, evaluate_model_by_weight, test_data, test_labels)
    sv_one = calculate_SV(permutations, dict_of_acc, 1, len(clientsToWeights))


    