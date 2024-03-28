from tensorflow.keras import layers, models
import logging
import tensorflow as tf
import numpy as np
import h5py
import asyncio
import aiohttp
from itertools import combinations
from app.utils import download_file_from_ipfs, download_model_from_ipfs, initialize_sample_model
from app.utils import download_weights_from_ipfs_async

########## Configurations ##########
TEST_DATA = "QmZxr5vzCirovKm2sGXqo1rHvC9mfsNeTMXjos9fcQri6X"
TEST_LABELS = "QmaqsdE1hrRWMRsV4eXpYfe4CnmALDLTaL97mVCnaZu1ST"
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
        # convert matrixSize and weights to numpy array
        if model_address:
            self.model = download_model_from_ipfs(model_address)
        else: # TODO: remove this else block
            self.model = initialize_sample_model()
        logging.info("Initialize the model successfully.")
        self.clientToWeights = asyncio.run(self.download_clients_weights(clients_submissions))
        self.download_test_data(test_data_add, test_labels_add)
        logging.info("Download the weights & data successfully.")
        self.evaluation_results = asyncio.run(self.main_evaluation())
        logging.info("Evaluation results are ready.")

    def get_evaluation_results(self):
        dummy_results = {
            "newModelAddress": "QmYEMkTVdYF7bBoJ28D2Lrqex1xozLZ5yHQ8pjDuJ18zQe",
            "clientIds": ["1", "2", "3", "4", "5"],
            "clientNewReputations": [100, 100, 100, 90, 80],
            "clientRewards": [100, 100, 100, 90, 80]
        }
        # results = {}
        # results["newModelAddress"] = self.newModelAddress
        # results["clientIds"] = self.evaluate_results.keys()
        # results["clientNewReputations"] = TODO
        # results["clientRewards"] = TODO
        return dummy_results
            
    def generate_weights_template(self, sample_weights):
        layers = []
        for i in sample_weights:
            layers.append(i.shape)
        return layers
    
    async def download_clients_weights(self, client_weights):
        """
        Convert client_id: weights_ipfs_address to client_id: weights_local_path
        """
        # client_weights : client_id: weights_address
        clientsToWeights = {}
        tasks = []
        async with aiohttp.ClientSession() as session:
            for client_id, weights_address in client_weights.items():
                task = asyncio.create_task(download_weights_from_ipfs_async(session, weights_address, client_id))
                tasks.append(task)
            paths = await asyncio.gather(*tasks)
        # return paths to client - path map
        for path in paths:
            clientsToWeights[path[0]] = path[1]
        return clientsToWeights

    def download_test_data(self, test_data_add, test_labels_add):
        # TODO: validation layer / error handling
        test_data_path = download_file_from_ipfs(test_data_add)
        test_labels_path = download_file_from_ipfs(test_labels_add)
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
        Evaluate the model with the given weights
        """
        model = self.model
        model.load_weights(weights_path)
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
    
    def get_all_combinations(self, clients_id):
        """
        Returns [["1", "2", "3], ["2", "3"]]...
        """
        all_combinations = []
        for i in range(1, len(clients_id)+1):
            all_combinations.extend(list(combinations(clients_id, i)))
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
    
    def calculate_avg_weights(self, dict_of_weights, tup):
        weights = [dict_of_weights[i] for i in tup]
        avg_weights = []
        weight_length = len(weights[0])
        for i in range(weight_length):
            single_sum = 0
            for weight in weights:
                single_sum += weight[i]
            avg_weights.append(single_sum / len(weights))
            # avg_weights.append(sum([weight[i] for weight in weights]) / len(weights))
        return avg_weights
    
    async def calculate_accuracy_for_permutations(self, permutations, dict_of_weights, initial_weights):
        """
        Input: a dictionary of permutations and a dictionary of weights
        Output: a dictionary of permutations and their corresponding accuracy
        """
        dict_of_acc = {}
        for tup in permutations:
            key = self.sort_tuple(tup)
            avg_weights = self.calculate_avg_weights(dict_of_weights, tup)
            acc = await self.evaluate_model
            dict_of_acc[key] = acc[1]
        return dict_of_acc

    
    def aggregate_new_models(self):
        """
        Aggregate the client weights to generate new global model
        """
        pass

### For Testing ###
# if __name__ == "__main__":
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
