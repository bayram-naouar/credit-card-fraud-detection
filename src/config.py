from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam
import numpy as np

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_RAW = ROOT_DIR / '..' /'data' / 'raw' / 'creditcard.csv'
DATA_PROCESSED_DIR = ROOT_DIR / '..' / 'data' / 'processed'
MODELS_DIR = ROOT_DIR / '..' / 'models'

class AutoEncoderBuilder:
    def __init__(self, input_layer_shape, **kwargs):
        self.latent_dim = kwargs.get("latent_dim", 4)
        self.learning_rate = kwargs.get("learning_rate", 0.001)
        self.loss = kwargs.get("loss", "mse")
        self.percentile = kwargs.get("percentile", 97)
        self.epochs = kwargs.get("epochs", 10)
        self.batch_size = kwargs.get("batch_size", 64)
        self.input_layer_shape = input_layer_shape
    
    def __call__(self):
        start_dim = 2**int(np.ceil(np.log2(self.input_layer_shape)))
        model = Sequential()
        model.add(Input(shape=(self.input_layer_shape,)))
        # Encoder
        dim = start_dim
        while dim > self.latent_dim:
            model.add(Dense(dim, activation='relu'))
            dim //= 2
        # Latent space
        model.add(Dense(self.latent_dim, activation='relu'))
        # Decoder
        dim = self.latent_dim * 2
        while dim <= start_dim:
            model.add(Dense(dim, activation='relu'))
            dim *= 2
        # Output layer
        model.add(Dense(self.input_layer_shape, activation='linear'))
        # Compile
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss=self.loss)
        return model