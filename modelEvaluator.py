from tensorflow.keras import layers, models
import logging
import tensorflow as tf
import numpy as np
import h5py
import asyncio
import aiohttp
from utils import ipfs_utils, ml_utils, logging_utils

########## Configurations ##########
TEST_DATA = "QmZxr5vzCirovKm2sGXqo1rHvC9mfsNeTMXjos9fcQri6X"
TEST_LABELS = "QmaqsdE1hrRWMRsV4eXpYfe4CnmALDLTaL97mVCnaZu1ST"
# MODEL = 
####################################

class MlModel:
    def __init__(self, client_weights, test_data_add=TEST_DATA, test_labels_add=TEST_LABELS, model_address=None):
        # convert matrixSize and weights to numpy array
        if model_address:
            self.model = ipfs_utils.download_model_from_ipfs(model_address)
        else: # TODO: remove this else block
            self.model = ml_utils.initialize_sample_model()
        logging.info("Initialize the model successfully.")
        self.clientToWeights = asyncio.run(self.download_clients_weights(client_weights))
        self.download_test_data(test_data_add, test_labels_add)
        logging.info("Download the weights & data successfully.")
        self.evaluation_results = asyncio.run(self.main_evaluation())
        logging.info("Evaluation results are ready.")

    def get_evaluation_results(self):
        cilentToResults = {}
        for tuple in self.evaluation_results:
            cilentToResults[tuple[0]] = tuple[1]
        return cilentToResults
            
    def generate_weights_template(self, sample_weights):
        layers = []
        for i in sample_weights:
            layers.append(i.shape)
        return layers
    
    async def download_clients_weights(self, client_weights):
        clientsToWeights = {}
        tasks = []
        async with aiohttp.ClientSession() as session:
            for i in range(len(client_weights)):
                task = asyncio.create_task(ipfs_utils.download_weights_from_ipfs_async(session, client_weights[i], i))
                tasks.append(task)
            paths = await asyncio.gather(*tasks)
        # return paths to client - path map
        for path in paths:
            clientsToWeights[path[0]] = path[1]
        return clientsToWeights

    def download_test_data(self, test_data_add, test_labels_add):
        # TODO: validation layer / error handling
        test_data_path = ipfs_utils.download_file_from_ipfs(test_data_add)
        test_labels_path = ipfs_utils.download_file_from_ipfs(test_labels_add)
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
        for client_id, weights_path in self.clientToWeights.items():
            task = asyncio.create_task(self.evaluate_model(weights_path, client_id))
            tasks.append(task)
        results = await asyncio.gather(*tasks)
        return results
    
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
