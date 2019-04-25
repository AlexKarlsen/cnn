from keras import Sequential
from keras.layers import Dropout, Dense
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input

# pylint: disable=fixme, relative-beyond-top-level
from .nn_base import nn_base

# TODO this should be renamed to somthing other than cnn
class inceptionresnetv2(nn_base):

    def __init__(
        self,
        pretrained_weights="imagenet",
        include_top=False,
        pooling = "max",
         **kwargs):

        self.pretrained_weights = pretrained_weights
        self.include_top = include_top
        self.pooling = pooling

        super().__init__(preprocessing_function=preprocess_input,**kwargs)

    def construct_model(self):
        self.base = InceptionResNetV2(input_shape=self.input_shape,
                                weights=self.pretrained_weights, 
                                include_top=self.include_top,
                                pooling=self.pooling)

        self.model = Sequential([
            self.base,
            Dropout(rate=0.8),
            Dense(self.n_classes, activation="softmax")
        ])

    def __str__(self):
        return f"inceptionresnetv2_{self.pretrained_weights}" 