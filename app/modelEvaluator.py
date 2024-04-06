from tensorflow.keras import layers, models
import logging
import tensorflow as tf
import numpy as np
import h5py
import math
import asyncio
import aiohttp
from itertools import combinations
from app.utils import download_file_from_ipfs, download_model_from_ipfs, initialize_sample_model
from app.utils import download_weights_from_ipfs_async
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
import os
import json

########## Configurations ##########
TEST_DATA = "QmZxr5vzCirovKm2sGXqo1rHvC9mfsNeTMXjos9fcQri6X"
TEST_LABELS = "QmaqsdE1hrRWMRsV4eXpYfe4CnmALDLTaL97mVCnaZu1ST"
_ = load_dotenv(find_dotenv())
FOLDER = os.getenv("TEMP_FOLDER")
TEMP_FILE_PATH = Path(__file__).parent / FOLDER
if not TEMP_FILE_PATH.exists():
    TEMP_FILE_PATH.mkdir()
####################################

# Example request body:
# {
#     "clientsToSubmissions":{"1":"QmXvmaD8FuPnySgNaxv3vun9ZtuMGdDFnNS6tsLKz8Jhyj","2":"QmXvmaD8FuPnySgNaxv3vun9ZtuMGdDFnNS6tsLKz8Jhyj","3":"QmXvmaD8FuPnySgNaxv3vun9ZtuMGdDFnNS6tsLKz8Jhyj"},
#     "clientsToReputation":{"1":"100","2":"100","3":"100"},
#     "rewardPool":"999999999999999900",
#     "modelAddress":"0x1234567890123456789012345678901234567890",
#     "testDataAddress":"0x1234567890123456789012345678901234567890",
#     "testLabelAddress":"0x1234567890123456789012345678901234567890",
#     "testDataHash":"0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
#     "testLabelHash":"0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"
# }

class MlModel:
    def __init__(self, clients_submissions, test_data_add=TEST_DATA, test_labels_add=TEST_LABELS, model_address=None, reward_pool=None, clients_reputation=None, test_data_hash=None, test_label_hash=None):
        """
        clients_submissions : dict
        """
        # convert matrixSize and weights to numpy array
        if model_address:
            self.model = download_model_from_ipfs(model_address)
        else: # TODO: remove this else block
            self.model = initialize_sample_model()
        logging.info("Initialize the model successfully.")
        self.clientToWeights = self.download_clients_weights(clients_submissions)
        self.download_test_data(test_data_add, test_labels_add)
        logging.info("Download the weights & data successfully.")
        self.dict_of_acc = self.calculate_accuracy_for_permutations(clients_submissions.keys(), self.clientToWeights)
        # contribution = self.calculate_federate_contribution(self.clientToWeights)
        # print(contribution)
        self.sv_results = self.calculate_SV_for_all_clients(self.permutations, self.clientToWeights)
        # print(sv_results)
        self.calculate_reputation(self.sv_results, reward_pool, clients_reputation)
        self.evaluation_res = self.get_evaluation_results()
        print(self.evaluation_res)
        # logging.info("Evaluation results are ready.")
    
    def calculate_reputation(self, sv_results, reward_pool, clients_reputation):
        """
        Calculate the reputation of the client and the reward
        """
        clientToRewards = {}
        sum_reputation_sv = sum([sv_results[client_id] * int(clients_reputation[client_id]) for client_id in sv_results.keys()])
        avg_sv = sum(sv_results.values()) / len(sv_results)
        for client_id, sv in sv_results.items():
            reputation = int(clients_reputation[client_id])
            reward = sv * reputation / sum_reputation_sv * float(reward_pool)
            if sv < avg_sv:
                reputation -= 1
                clients_reputation[client_id] = str(reputation)
            clientToRewards[client_id] = round(reward, 0)
        self.clientToRewards = clientToRewards
        self.clientToReputation = clients_reputation

        
    def get_evaluation_results(self):
        # dummy_results = {
        #     "newModelAddress": "QmYEMkTVdYF7bBoJ28D2Lrqex1xozLZ5yHQ8pjDuJ18zQe",
        #     "clientIds": ["1", "2", "3", "4", "5"],
        #     "clientNewReputations": [100, 100, 100, 90, 80],
        #     "clientRewards": [100, 100, 100, 90, 80]
        # }
        results = {}
        results["newModelAddress"] = "QmPvZbSMP7sfDhdsEFt5AWmjs67LUKoXjLu46q29cJ4CWL"
        results["clientIds"] = list(self.clientToWeights.keys())
        results["clientNewReputations"] = json.loads(json.dumps(self.clientToReputation))
        results["clientRewards"] = json.loads(json.dumps(self.clientToRewards))
        return results
            
    def generate_new_model(self, clientToWeights):
        """
        Generate a new model with the given weights
        """
        avg_weights = self.calculate_avg_weights(clientToWeights, clientToWeights.keys())
        model = self.model
        model.set_weights(avg_weights)
        model.save_weights(TEMP_FILE_PATH / "new_weights.h5")
        
    
    def download_clients_weights(self, client_weights):
        """
        Convert client_id: weights_ipfs_address to client_id: weights_local_path
        """
        # client_weights : client_id: weights_address
        clientsToWeights = {}
        # tasks = []
        # async with aiohttp.ClientSession() as session:
        for client_id, weights_address in client_weights.items():
            path = download_file_from_ipfs(weights_address, client_id)
            clientsToWeights[path[0]] = path[1]
        #         task = asyncio.create_task(download_file_from_ipfs(session, weights_address, client_id))
        #         tasks.append(task)
        #     paths = await asyncio.gather(*tasks)
        # return paths to client - path map
        # for path in paths:
        #     clientsToWeights[path[0]] = path[1]
        return clientsToWeights

    def download_test_data(self, test_data_add, test_labels_add):
        # TODO: validation layer / error handling
        test_data_path = download_file_from_ipfs(test_data_add, 0)[1]
        test_labels_path = download_file_from_ipfs(test_labels_add, 0)[1]
        with h5py.File(test_data_path, "r") as hf:
            data = hf["test_data"][:]
            hf.close()
        with h5py.File(test_labels_path, "r") as hf:
            labels = hf["test_labels"][:]
            hf.close()
        self.test_data = data
        self.test_labels = labels
        logging.info(f"Test data and labels downloaded successfully in the shape: {self.test_data.shape, self.test_labels.shape}")

    async def evaluate_model(self, weights_path, client_id):
        """
        Evaluate the model by the weights path
        """
        model = self.model
        model.load_weights(weights_path)
        evaluation_results = model.evaluate(self.test_data, self.test_labels)
        return client_id, evaluation_results
    
    def evaluate_model_by_weight(self, weights, client_id):
        """
        Evaluate the model by the weights
        """
        model = self.model
        model.load_weights(weights)
        evaluation_results = model.evaluate(self.test_data, self.test_labels)
        return client_id, evaluation_results
    
    async def main_evaluation(self):
        """
        Evaluate the model with the given weights
        """
        tasks = []
        all_weights_paths = self.clientToWeights.values()
        for client_id, weights_path in self.clientToWeights.items():
            task = asyncio.create_task(self.evaluate_model(weights_path, client_id))
            tasks.append(task)
        results = await asyncio.gather(*tasks)
        return results
    
    def get_all_combinations(self, clients_ids):
        """
        Input: list of clients' ids
        For each possible permutation length, generate permutations
        Returns [["1", "2", "3], ["2", "3"]]...
        """
        all_combinations = []
        for i in range(1, len(clients_ids)+1):
            all_combinations.extend(list(combinations(clients_ids, i)))
        return all_combinations
    
    def sort_tuple(self, tup):
        """
        Input: a tuple
        Output: a sorted string
        """
        result = list(tup)
        result.sort()
        str_result = ''.join(list(map(lambda x: str(x), result)))
        return str_result
    
    def calculate_avg_weights(self, clientToWeights, tup):
        weights = []
        for i in tup:
            # Only load the weights in client tup
            model = self.model
            model.load_weights(clientToWeights[i])
            weights.append(model.get_weights())
        avg_weights = []
        for i in range(len(weights[0])):
            avg_weights.append(np.average([model[i] for model in weights], axis=0))
        return avg_weights
    
    def calculate_accuracy_for_permutations(self, clients_ids, clientToWeights):
        """
        Input: a dictionary of permutations and a dictionary of weights
        Output: a dictionary of permutations and their corresponding accuracy
        """
        dict_of_acc = {}
        self.permutations = self.get_all_combinations(clients_ids)
        for tup in self.permutations:
            key = self.sort_tuple(tup)
            avg_weights = self.calculate_avg_weights(clientToWeights, tup)
            model = self.model
            model.set_weights(avg_weights)
            model.save_weights(TEMP_FILE_PATH / "temp_weights.h5")
            acc = self.evaluate_model_by_weight(TEMP_FILE_PATH / "temp_weights.h5", key)
            dict_of_acc[key] = acc[1][1]
        # print(dict_of_acc)
        dict_of_acc[''] = self.model.evaluate(self.test_data, self.test_labels)[1]
        return dict_of_acc
    
    def calculate_SV_for_all_clients(self, permutations, clientToWeights):
        """
        Input: a list of permutations and a dictionary of weights
        Output: a dictionary of clients and their corresponding SV
        """
        dict_of_SV = {}
        N = len(clientToWeights)
        for client_id in clientToWeights.keys():
            SV = self.calculate_SV(permutations, client_id, N)
            dict_of_SV[client_id] = SV
        return dict_of_SV

    def calculate_SV(self, permutations, client_id, N, if_logistic=True):
        print("Calculating SV for client {}".format(client_id))
        SV = 0
        for tup in permutations:
            if client_id in tup:
                S = len(tup)
                tag = self.sort_tuple(tup)
                current_acc = self.dict_of_acc[tag]
                minusset = set(tup) - set([client_id])
                minusset_tag = self.sort_tuple(minusset)
                minusset_acc = self.dict_of_acc[minusset_tag]
                # print("minusset: {}".format(minusset_acc))
                weight = self.calculate_weight(N, S)
                SV += weight * self.logistic_function((current_acc - minusset_acc))
        print("SV for client {} is {}".format(client_id, SV))
        return SV
    
    def logistic_function(self, x):
        return 1 / (1 + math.exp(-x))

    def calculate_weight(self, N, S):
        """
        Input: number of clients, number of clients in a subset
        Output: weight
        """
        return math.factorial(N - S) * math.factorial(S - 1) / math.factorial(N)
    
    def calculate_federate_contribution(self, clientToWeights):
        tup = self.sort_tuple(clientToWeights.keys())
        global_weight = self.calculate_avg_weights(clientToWeights, tup)
        clientToContribution = {}

        for client_id, weights_path in clientToWeights.items():
            model = self.model
            model.load_weights(weights_path)
            client_weight = model.get_weights()
            federate_contribution = []
            for i in range(len(global_weight)):
                federate_contribution.append(np.linalg.norm(global_weight[i] - client_weight[i]))
            # 2th norm
            federate_contribution_score = np.linalg.norm(federate_contribution)
            print("Federate contribution for client {} is {}".format(client_id, federate_contribution_score))
            clientToContribution[client_id] = federate_contribution_score
        return clientToContribution

        


        

### For Testing ###
if __name__ == "__main__":
#     loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#     model = models.Sequential([
#         layers.Flatten(input_shape=(32, 32)),
#         layers.Dense(128, activation="relu"),
#         layers.Dense(64, activation="relu"),
#         layers.Dropout(0.2),
#         layers.Dense(10)
#     ])
#     model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
#     sample_weights = model.get_weights()
#     client_weights = [ "QmXvmaD8FuPnySgNaxv3vun9ZtuMGdDFnNS6tsLKz8Jhyj", "QmXvmaD8FuPnySgNaxv3vun9ZtuMGdDFnNS6tsLKz8Jhyj" ]
#     test = MlModel(client_weights)
    example = {
        "clientsToSubmissions":{"1":"QmdfzatZMtMmaWMTNsJuvCQcBbHAQFSGTrYLi6CiZ5fWTi","2":"QmX4T5dLBrubvGmUJnkM1tP1j8oPC5ZwDMUWdDR1R9wWNN","3":"QmTCxkXPG9fetQs1mP6QmQDWAx5vUc325u8AndPquJQv62","4":"QmPZeG6uJsX1EsKd4BeR5cPNBScAwpU7CLXRGXrEgbUpSV","5":"Qmc8VCb4DWZbXxhqcCCsqdVoYiGqrmkVbQ4fHPhYopRwS1"},
        "clientsToReputation":{"1":"100","2":"100","3":"100", "4": "100", "5": "100"},
        "rewardPool":"999999999999999900",
        "modelAddress":"QmYEMkTVdYF7bBoJ28D2Lrqex1xozLZ5yHQ8pjDuJ18zQe",
        "testDataAddress":"Qmeo6yw83vLYX3zkgPDiGK24FB99pz1PX1EngLHLPyko76",
        "testLabelAddress":"QmWTBKyjTsuHHq6kHdCnbvF3PQZDQa2pincoyAc8DzisFX",
        "testDataHash":"64d59ad605ca4af2b947844984c54f11409928c2ad9880f864ee7459bb17e308",
        "testLabelHash":"ae416418d6466f81e7a63f0d8c09400fa221c6389eca6c67d7b12370715c385c"
    }
    test = MlModel(example["clientsToSubmissions"], example["testDataAddress"], example["testLabelAddress"], example["modelAddress"], example["rewardPool"], example["clientsToReputation"], example["testDataHash"], example["testLabelHash"])

    
