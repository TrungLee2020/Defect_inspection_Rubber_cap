import os
from keras.preprocessing.image import ImageDataGenerator
import configCAE

class Preprocessor:
    def __init__(self, input_directory, rescale, shape, color_mode, preprocessing_function):
        self.input_directory = input_directory
        self.train_data_dir = os.path.join(input_directory, "train")
        self.test_data_dir = os.path.join(input_directory, "test")
        self.rescale = rescale
        self.shape = shape
        self.color_mode = color_mode
        self.preprocessing_function = preprocessing_function
        self.validation_split = configCAE.VAL_SPLIT

        self.nb_val_images = None
        self.nb_test_images = None

    def get_train_generator(self, batch_size, shuffle=True):
        # preprocessing and realtime data augmentation:
        train_datagen = ImageDataGenerator(
            # standard input
            featurewise_center=False,
            featurewide_std_normalization=False,
            # rotate
            rotation_range=configCAE.ROT_ANGLE,
            # horinzontally
            width_shift_range = configCAE.W_SHIFT_RANGE,
            # vertically
            height_shift_range = configCAE.config.H_SHIFT_RANGE,
            # filling points outside the input boundaries
            fill_mode = configCAE.config.FILL_MODE,
            # value used for fill_mode = 'constant'
            cval = 0.0,
            # random change brightness (darker < 1 < brighter)
            brightness_range = configCAE.BRIGHTNESS_RANGE,
            # scale factor
            rescale = self.rescale,
            # set function that will be applied on each input
            preprocessing_function = self.preprocessing_function,
            # image data format, either "channels_first" or "channels_last"
            data_format = "channel_last",
            # valdation
            validation_split = self.validation_split,
        )
    
        # Generate training batchs with datagen.flow_from_directory()
        train_generator = train_datagen.flow_from_directory(
            directory = self.train_data_dir,
            target_size = self.shape,
            color_mode = self.color_mode,
            batch_size = batch_size,
            class_mode = "input",
            subset = "training",
            shuffle = True,
        )

        return train_generator
    
    def get_val_generator(self, batch_size, shuffle=True):
        """
        For training, pass autoencoder.batch_size as batch size
        For validation, pass autoencoder.batch_size as batch size
        for test, pass nb_test_images as batch size
        """
        # For validation dataset, only rescaling
        validation_datagen = ImageDataGenerator(
            rescale=self.rescale,
            data_format="channel_last",
            validation_split = self.validation_split,
            preprocessing_function = self.preprocessing_function,
        )

        # Generate validation batchs with datagen.flow_from_firectory()
        validation_generator = validation_datagen.flow_from_directory(
            directory=self.train_data_dir,
            target_size = self.shape,
            color_mode = self.color_mode,
            batch_size = batch_size,
            class_mode = "input",
            subset="validation",
            shuffle=shuffle,
        )

        return validation_generator
    
    def get_test_generator(self, batch_size, shuffle=False):
        """
        For training, pass autoencoder.batch_size as batch size.
        For validation, pass nb_validation_images as batch size.
        For test, pass nb_test_images as batch size.
        """
        # For test dataset, only rescaling
        test_datagen = ImageDataGenerator(
            rescale=self.rescale,
            data_format="channels_last",
            preprocessing_function=self.preprocessing_function,
        )

        # Generate validation batches with datagen.flow_from_directory()
        test_generator = test_datagen.flow_from_directory(
            directory=self.test_data_dir,
            target_size=self.shape,
            color_mode=self.color_mode,
            batch_size=batch_size,
            class_mode="input",
            shuffle=shuffle,
        )
        return test_generator

    def get_finetuning_generator(self, batch_size, shuffle=False):
        """
        For training, pass autoencoder.batch_size as batch size.
        For validation, pass nb_validation_images as batch size.
        For test, pass nb_test_images as batch size.
        """
        # For test dataset, only rescaling
        test_datagen = ImageDataGenerator(
            rescale=self.rescale,
            data_format="channels_last",
            preprocessing_function=self.preprocessing_function,
        )

        # Generate validation batches with datagen.flow_from_directory()
        finetuning_generator = test_datagen.flow_from_directory(
            directory=self.test_data_dir,
            target_size=self.shape,
            color_mode=self.color_mode,
            batch_size=batch_size,
            class_mode="input",
            shuffle=shuffle,
        )
        return finetuning_generator
    
    def get_total_number_test_images(self):
        total_number = 0
        sub_dir_names = os.listdir(self.test_data_dir)
        for sub_dir_name in sub_dir_names:
            sub_dir_path = os.path.join(self.test_data_dir, sub_dir_name)
            filenames = os.listdir(sub_dir_path)
            number = len(filenames)
            total_number = total_number + number
        return total_number
    
    def get_preprocessing_function(architecture):
        if architecture in ["mvtecCAE", "baselineCAE", "indexptionCAE", "resnetCAE"]:
            preprocessing_function = None
        return preprocessing_function
