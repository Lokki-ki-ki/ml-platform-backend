import logging
import numpy as np
import tensorflow as tf
import asyncio
from tensorflow.keras import layers, models

def initialize_sample_model(size_of_label: int = 10): 
    # Use mlp to represent the model
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model = models.Sequential([
        layers.Flatten(input_shape=(32, 32)),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(size_of_label)
    ])
    model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
    return model

def initialize_testing_data(test_size=100, input_shape=(32, 32)): 
    return np.random.rand(test_size, 32, 32), np.random.randint(0, 10, size=input_shape)

def evaluate_model(weights_path, client_id, sample_model, test_data, test_labels):
    """
    Evaluate the model by the weights path
    """
    model = sample_model
    model.load_weights(weights_path)
    evaluation_results = model.evaluate(test_data, test_labels)
    return client_id, evaluation_results

def evaluate_model_by_weight(weights, client_id, sample_model, test_data, test_labels):
    """
    Evaluate the model by the weights
    """
    model = sample_model
    model.load_weights(weights)
    evaluation_results = model.evaluate(test_data, test_labels)
    return client_id, evaluation_results

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

async def evaluate_model(weights_path, client_id, sample_model, test_data, test_labels):
    """
    Evaluate the model by the weights path
    """
    model = sample_model
    model.load_weights(weights_path)
    evaluation_results = model.evaluate(test_data, test_labels)
    return client_id, evaluation_results
    
async def main_evaluation(clientToWeights):
    """
    Evaluate the model with the given weights
    """
    tasks = []
    for client_id, weights_path in clientToWeights.items():
        task = asyncio.create_task(evaluate_model(weights_path, client_id))
        tasks.append(task)
    results = await asyncio.gather(*tasks)
    return results

