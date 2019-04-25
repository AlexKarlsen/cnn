from keras import Sequential
from keras.layers import Dense
from keras.applications.resnet50 import ResNet50, preprocess_input

# pylint: disable=fixme, relative-beyond-top-level
from .nn_base import nn_base

class resnet50(nn_base):

    def __init__(
        self,
        pooling="avg",
        include_top = False,
        pretrained_weights = "imagenet",
         **kwargs):

        self.pooling = pooling
        self.include_top = include_top
        self.pretrained_weights = pretrained_weights

        super().__init__(preprocessing_function=preprocess_input,**kwargs)

    def construct_model(self):

        self.base = ResNet50(input_shape=self.input_shape,
                                weights=self.pretrained_weights, 
                                include_top=self.include_top,
                                pooling=self.pooling)

        self.model = Sequential([
            self.base,
            Dense(self.n_classes, activation="softmax")
        ])


    def __str__(self):
        return f"resnet50_{self.pretrained_weights}" 