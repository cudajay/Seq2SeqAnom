import DataStore
import tensorflow as tf


class Experiment():
    def __init__(self,prototype:tf.keras.Model, model_keys:{}):
        self.datastore: DataStore.DataStore
        self.params: {}
        self.models: {}
        self.results: {}

    def run(self):
        pass


