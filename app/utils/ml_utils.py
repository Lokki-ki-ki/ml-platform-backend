import logging
import numpy as np
import tensorflow as tf
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

