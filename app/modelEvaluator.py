from tensorflow.keras import layers, models
import logging
import tensorflow as tf
import numpy as np
from itertools import combinations
from app.utils import download_file_from_ipfs, download_model_from_ipfs, initialize_sample_model, download_test_data, formatting_weights, evaluate_model_by_weight, calculate_avg_weights, upload_to_ipfs
from app.shapley import calculate_SV, prepare_SV
from app.shapley import truncated_monte_carlo_shapley
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
        self.sample_weights = self.model.get_weights()
        self.clientToWeights = formatting_weights(self.download_clients_weights(clients_submissions))
        self.clientReputation = formatting_weights(clients_reputation)
        self.test_data, self.test_labels = download_test_data(test_data_add, test_labels_add)
        logging.info("Download the weights & data successfully.")
        self.sv_results = self.calculate_SV_for_all_clients(self.clientToWeights, self.test_data, self.test_labels)
        self.calculate_reputation(self.sv_results, reward_pool, self.clientReputation)
        self.new_model = self.generate_new_model(self.clientToWeights, self.model)
        self.evaluation_res = self.get_evaluation_results()
        print(self.evaluation_res)
        # self.monte_carlo = self.calculate_SV_by_Monte_Carlo_for_all_clients(self.model, self.sample_weights, self.test_data, self.test_labels, evaluate_model_by_weight, self.clientToWeights)
        # print(self.monte_carlo, 'monte_carlo')
        # print(self.sv_results, 'sv_results')
    
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
        results["newModelAddress"] = self.new_model
        results["clientIds"] = list(self.clientToWeights.keys())
        results["clientNewReputations"] = json.loads(json.dumps(self.clientToReputation))
        results["clientRewards"] = json.loads(json.dumps(self.clientToRewards))
        return results
            
    def generate_new_model(self, clientToWeights, sample_model):
        """
        Generate a new model with the given weights
        """
        avg_weights = calculate_avg_weights(clientToWeights, clientToWeights.keys(), sample_model)
        model = self.model
        model.set_weights(avg_weights)
        model.save_weights(TEMP_FILE_PATH / "new_weights.h5")
        new_cid = upload_to_ipfs(TEMP_FILE_PATH / "new_weights.h5")
        os.rename(TEMP_FILE_PATH / "new_weights.h5", TEMP_FILE_PATH / f"{new_cid}.h5")
        return new_cid
        
    def download_clients_weights(self, client_weights):
        """
        Convert client_id: weights_ipfs_address to client_id: weights_local_path
        """
        clientsToWeights = {}
        for client_id, weights_address in client_weights.items():
            path = download_file_from_ipfs(weights_address, client_id)
            clientsToWeights[path[0]] = path[1]
        return clientsToWeights
    
    def calculate_SV_for_all_clients(self, clientToWeights, test_data, test_labels):
        """
        Input: a list of permutations and a dictionary of weights
        Output: a dictionary of clients and their corresponding SV
        """
        permutations, dict_of_acc = prepare_SV(clientToWeights, self.model, self.sample_weights, evaluate_model_by_weight, test_data, test_labels)
        # sv_one = calculate_SV(permutations, dict_of_acc, 1, len(clientsToWeights))

        dict_of_SV = {}
        N = len(clientToWeights)
        for client_id in clientToWeights.keys():
            SV = calculate_SV(permutations, dict_of_acc, client_id, N)
            dict_of_SV[client_id] = SV
        return dict_of_SV
    
    def calculate_federate_contribution(self, clientToWeights):
        tup = self.sort_tuple(clientToWeights.keys())
        global_weight = calculate_avg_weights(clientToWeights, tup)
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
    
    def calculate_SV_by_Monte_Carlo_for_all_clients(self, sample_model, sample_weights, test_data, test_labels, evaluate_model_by_weight, clientsToWeight):
        """
        Calculate Shapley Value by Monte Carlo for all clients
        """
        res = truncated_monte_carlo_shapley(sample_model, sample_weights, test_data, test_labels, evaluate_model_by_weight, clientsToWeight)
        return res


### For Testing ###
if __name__ == "__main__":
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

    
