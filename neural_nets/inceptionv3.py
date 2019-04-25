from keras import Sequential
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.layers import Dense, Dropout

# pylint: disable=fixme, relative-beyond-top-level
from .nn_base import nn_base

# TODO this should be renamed to somthing other than cnn
class inceptionv3(nn_base):

    def __init__(
        self,
        pretrained_weights="imagenet",
        include_top=False,
        pooling = "avg",
         **kwargs):

        self.pooling = pooling
        self.include_top = include_top
        self.pretrained_weights = pretrained_weights

        super().__init__(preprocessing_function=preprocess_input, **kwargs)

    def construct_model(self):
        self.base = InceptionV3(
            input_shape=self.input_shape,
            weights=self.pretrained_weights, 
            include_top=self.include_top,
            pooling=self.pooling)

        self.model = Sequential([
            self.base,
            Dense(1024, activation='relu'),
            Dropout(rate=0.6),
            Dense(self.n_classes, activation="softmax")
        ])

    def __str__(self):
        return f"inceptionv3{self.pretrained_weights}" 