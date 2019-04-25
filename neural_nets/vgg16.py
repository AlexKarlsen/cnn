from keras.applications.vgg16 import VGG16, preprocess_input
from keras import Sequential
from keras.layers import Dense

# pylint: disable=fixme, relative-beyond-top-level
from .nn_base import nn_base

# TODO this should be renamed to somthing other than cnn
class vgg16(nn_base):

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
        # Get VGG16 base i.e. feature extractor part
        # base is public to enable fine-tuning i.e. freezing
        self.base = VGG16(
            include_top=self.include_top,
            input_shape=self.input_shape,
            weights=self.pretrained_weights,
            pooling=self.pooling
            )

        # Adding the classification layer i.e. softmax
        self.model = Sequential([
            self.base,
            Dense(4096, activation='relu'),
            Dense(4096, activation='relu'),
            Dense(self.n_classes, activation="softmax")
        ])


    def __str__(self):
        return f"vgg16"
        #TODO add any significant parameters to name


