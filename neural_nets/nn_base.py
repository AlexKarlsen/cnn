from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, Callback
from keras.optimizers import Adam
import os
from os.path import realpath, dirname, join, exists
from os import makedirs
import pandas as pd
import numpy as np


from abc import ABCMeta, abstractmethod



class fine_tuining_callback(Callback):
    """
    Callback class used to configure fine tuning behavior
    """
    def __init__(self,model_ref,start,tune_for,**kwargs):
        self.start = start
        self.end = start+tune_for
        self.model_ref = model_ref
    
    def on_epoch_begin(self,epoch,logs={}):
        if epoch == self.start:
            self.model_ref.activate_tuning()

        elif epoch == self.end:
            self.model_ref.deactivate_tuning()
            

# TODO this should be renamed to somthing other than cnn
class nn_base(metaclass = ABCMeta):

    def __init__(
        self,
        n_classes,
        input_shape,
        start_timestamp,
        tuning_params = {},
        save_checkpoints = False,
        early_stopping = False,
        dataset_name = "unknown_dataset",
        runtime_augmentation = {},
        verbose = 1,
        preprocessing_function = None,
        batch_size=32,
        epochs=5,
        enable_tensorboard = True,
        **kwargs):

        self.start_timestamp = start_timestamp
        self.save_checkpoints = save_checkpoints
        self.early_stopping = early_stopping
        self.enable_tensorboard = enable_tensorboard
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.epochs = epochs
        self.dataset_name = dataset_name
        self.verbose = verbose
        self.tuning_params = tuning_params

        # is initialized by concrete subclass
        self.model = None
        self.base = None

        # setup data generation stuff
        gen_args_train = dict(
            preprocessing_function = preprocessing_function
        )

        gen_args_predict = gen_args_train


        # ensure that validation generator does not augment!
        #self.gen_train = ImageDataGenerator(**gen_args_train, **runtime_augmentation)
        #self.gen_predict = ImageDataGenerator(**gen_args_predict)

        self.train_datagen = ImageDataGenerator(**gen_args_train, **runtime_augmentation
                                                              #rotation_range=360,
                                                              #width_shift_range=0.2,
                                                              #height_shift_range=0.2,
                                                              #shear_range=10,
                                                              #zoom_range=0.2,
                                                              #horizontal_flip=True,
                                                              #vertical_flip=True
                                                            )
        # Rescale all images by 1./255 and apply image augmentation


        self.validation_datagen = ImageDataGenerator(**gen_args_predict)

        self.test_datagen = ImageDataGenerator(**gen_args_predict)

        self.define_callbacks()

        # construct the model using the function defined by the concrete model
        # after this call the model should be ready to be compiled
        self.construct_model()

        # freeze base if pretraining is used
        if not self.base is None:
            self.base.trainable = False

        self.compile()
       
    def compile(self, lr=0.01):

        self.model.compile(
            optimizer=Adam(lr=lr),
            loss='categorical_crossentropy',
            metrics=['categorical_accuracy']
            )

    def activate_tuning(self):
        _apply_tuning_params_to_model(self.base,**self.tuning_params)
        self.compile(lr=0.001)
        # workaround for warning, since its not compiled model will still be trainable
        # https://github.com/tensorflow/tensorflow/issues/22012
        self.base.trainable = True
    
    def deactivate_tuning(self):       
        self.base.trainable = False
        self.compile()

    def define_callbacks(self):

        self.callbacks = []

        cur_dir = dirname(realpath(__file__))
        path_model = join(cur_dir,"saved_models",self.dataset_name,f"{self}.h5")
        dir_model = dirname(path_model)
        # we store each log in seperate folder named after the classifier
        dir_tensor_board_logs = join(cur_dir, "tensor_board_logs", self.start_timestamp, self.dataset_name, f"{self}","")


        if not exists(dir_model): makedirs(dir_model)
        if not exists(dir_tensor_board_logs) : makedirs(dir_tensor_board_logs)

        if self.save_checkpoints:
            save_best = ModelCheckpoint(
                filepath=path_model,
                verbose=self.verbose,
                save_best_only=True,
                save_weights_only=False,
                mode='auto',
                period=1)

            self.callbacks.append(save_best)

        if self.enable_tensorboard:
            tb = TensorBoard(
                log_dir=dir_tensor_board_logs,
                batch_size=self.batch_size)
            self.callbacks.append(tb)

        if self.early_stopping:
            early = EarlyStopping(verbose=self.verbose)
            self.callbacks.append(early)

        if not self.tuning_params == {}:
            tune = fine_tuining_callback(self,**self.tuning_params)
            self.callbacks.append(tune)

    def append_ext(self, fn):
        return 'Image'+str(fn)+'.jpg'

    def train(self):
        data_dir = 'data'
        train_dir = 'Train'
        validation_dir = 'Validation'
        test_dir = 'Test'
        train_image_dir = join(data_dir,train_dir, 'TrainImages')
        validation_image_dir = join(data_dir, validation_dir,'./ValidationImages')
        self.test_image_dir = join(data_dir, test_dir,'./TestImages')

        y_train = pd.read_csv(join(data_dir, train_dir, 'trainLbls.csv'),names = ['label'])
        y_train -= 1
        y_train['label'] = y_train['label'].astype(str)
        y_train['id'] = pd.Series(np.arange(1,len(y_train)+1))

        y_validation = pd.read_csv(join(data_dir, validation_dir, 'valLbls.csv'), names = ['label'])
        y_validation -= 1
        y_validation['label'] = y_validation['label'].astype(str)
        y_validation['id'] = pd.Series(np.arange(1,len(y_validation)+1))

        #cat_y_train = to_categorical(y_train['label'])
        #cat_y_validation = to_categorical(y_validation['label'])

        
  
        y_train['id']=y_train['id'].apply(self.append_ext)
        y_validation['id']=y_validation['id'].apply(self.append_ext)

        x_test = pd.DataFrame()
        x_test['id'] = pd.Series(np.arange(1,3461))
        x_test['id'] = x_test['id'].apply(self.append_ext)
        #labels_training = to_categorical(labels_training)
        #labels_validation = to_categorical(labels_validation)

        # we fit on the same data to ensure that the same transformation is used
        #self.gen_train.fit(data_training)
        #self.gen_predict.fit(data_training)

        #iter_train = self.gen_train.flow(data_training,labels_training,self.batch_size)
        #iter_val = self.gen_predict.flow(data_validation,labels_validation,self.batch_size)

        #steps_per_epoch = data_training.shape[0] // self.batch_size
        #validation_steps = data_validation.shape[0] // self.batch_size


        # Flow training images in batches of 20 using train_datagen generator
        train_generator = self.train_datagen.flow_from_dataframe(
                        y_train,
                        train_image_dir,  # Source directory for the training images 
                        x_col="id",
                        y_col="label",
                        target_size=(256, 256),  
                        batch_size=self.batch_size)

        # Flow validation images in batches of 20 using test_datagen generator
        validation_generator = self.validation_datagen.flow_from_dataframe(
                y_validation,
                validation_image_dir, # Source directory for the validation images 
                x_col="id",
                y_col="label",
                target_size=(256, 256),
                batch_size=self.batch_size)

        steps_per_epoch = train_generator.n // self.batch_size
        validation_steps = validation_generator.n // self.batch_size

        self.model.fit_generator(train_generator, 
                              steps_per_epoch = steps_per_epoch,
                              epochs=self.epochs, 
                              workers=4,
                              validation_data=validation_generator, 
                              validation_steps=validation_steps,
                              callbacks = self.callbacks)

    def predict(self,data):
        x_test = pd.DataFrame()
        x_test['id'] = pd.Series(np.arange(1,3461))
        x_test['id'] = x_test['id'].apply(self.append_ext)

        test_generator = self.test_datagen.flow_from_dataframe(
                x_test,
                self.test_image_dir, # Source directory for the test images 
                x_col="id",
                class_mode=None,
                shuffle=False,
                target_size=(256, 256),
                batch_size=self.batch_size)
        # Do not shuffle!
        labels = self.model.test_generator(test_generator).argmax(axis=1)

        return labels

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def construct_model(self):
        """
        defines the model of the classifier.
        """
        pass

def _apply_tuning_params_to_model(
    model,
    trainable_base = None,
    trainable_fraction = None,
    trainable_layers = None,
    **kwargs):
        
        if not trainable_base is None:
            model.trainable = trainable_base
            return

        if not trainable_fraction is None:
            n_layers = round(len(model.layers) * trainable_fraction)
            for l in model.layers[:-n_layers]:
                l.trainable = True
            return

        if not trainable_layers is None:
            for l in model.layers[trainable_layers:]:
                l.trainable = True
            return