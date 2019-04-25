from keras.applications.vgg19 import VGG19, preprocess_input
from keras import Sequential
from keras.layers import Dense

# pylint: disable=fixme, relative-beyond-top-level
from .nn_base import nn_base

# TODO this should be renamed to somthing other than cnn
class vgg19(nn_base):

    def __init__(
        self,
        pooling="avg",
        include_top = False,
        save_best = False,
        early_stopping = False,
        pretrained_weights = "imagenet",
        **kwargs):

        self.pooling = pooling
        self.include_top = include_top
        self.save_best = save_best
        self.early_stopping = early_stopping
        self.pretrained_weights = pretrained_weights

        super().__init__(preprocessing_function=preprocess_input,**kwargs)

        
    def construct_model(self):
        # Get VGG16 base i.e. feature extractor part
        # base is public to enable fine-tuning i.e. freezing
        self.base = VGG19(
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
        return f"vgg19"
        #TODO add any significant parameters to name


