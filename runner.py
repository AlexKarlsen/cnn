import pandas as pd
import os
import numpy as np
import time

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))

from neural_nets.inceptionv3 import inceptionv3
from neural_nets.inceptionresnetv2 import inceptionresnetv2

parameters = {
                "dataset_name":"fin-benthic",
                "epochs" : 60,
                "batch_size": 64,
                "save_best": True,
                "pooling": 'avg',
                "tuning_params" : {
                    "start": 5,
                    "tune_for": 50,
                    "trainable_layers": 600
                },
                "runtime_augmentation" : {
                    #"rescale":1./255,
                    "rotation_range":360,
                    "width_shift_range":0.2,
                    "height_shift_range":0.2,
                    #shear_range=10,
                    "zoom_range":0.2,
                    "horizontal_flip":True,
                    "vertical_flip":True
                }
            }

model = inceptionresnetv2(n_classes=29, input_shape = (256, 256, 3),start_timestamp = str(time.time()), **parameters)

model.train()