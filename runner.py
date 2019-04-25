import pandas as pd
import os
import numpy as np
import time

from neural_nets.resnet50 import resnet50

parameters = {
                "epochs" : 15,
                "batch_size": 12,
                "save_best": True,
                "runtime_augmentation" : {
                    "rotation_range": 0.2,
                    "height_shift_range": 0.2,
                    "width_shift_range": 0.2,
                    "zoom_range": 0.2,
                    "horizontal_flip": True,
                    "vertical_flip": True
                }
            }

model = resnet50(n_classes=29, input_shape = (256, 256, 3),start_timestamp = str(time.time()), **parameters)

model.train()