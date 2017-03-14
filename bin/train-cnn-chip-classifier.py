# gbdx.Task(train-cnn-chip-classifier, chip_dir, min_side_dim, max_side_dim, resize_dim, hyperparams)

# For now we use the geojson that is present in the output chip directory. User does not input the geojson with class names to the train task (classes are defined when chips are generated).

import logging
import ast, subprocess
import shutil, os, sys
import geojson, json
import numpy as np
import geojsontools as gt

from sklearn.metrics import classification_report
from multiprocessing import Pool, cpu_count
from functools import partial
from osgeo import gdal
from scipy.misc import imresize
from gbdx_task_interface import GbdxTaskInterface
from keras.models import Sequential
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Dense, Dropout, Activation, Flatten


# log file for debugging
logging.basicConfig(filename='out.log',level=logging.DEBUG)


class TrainCnnChipClassifier(GbdxTaskInterface):

    def __init__(self):

        GbdxTaskInterface.__init__(self)
        # self.check_chip = check_chip

        # Get input string ports
        self.max_side_dim = int(self.get_input_string_port('max_side_dim', default='150'))
        self.min_side_dim = int(self.get_input_string_port('min_side_dim', default='0'))
        self.classes = [i.strip() for i in self.get_input_string_port('classes').split(',')]
        self.resize_dim = ast.literal_eval(self.get_input_string_port('resize_dim', default='None'))
        self.two_rounds = ast.literal_eval(self.get_input_string_port('two_rounds', default='False'))
        self.train_size = int(self.get_input_string_port('train_size', default='10000'))
        self.train_size_2 = int(self.get_input_string_port('train_size_2', default=int(0.5 * self.train_size)))
        self.batch_size = int(self.get_input_string_port('batch_size', default='32'))
        self.nb_epoch = int(self.get_input_string_port('nb_epoch', default='35'))
        self.nb_epoch_2 = int(self.get_input_string_port('nb_epoch_2', default='8'))
        self.use_lowest_val_loss = ast.literal_eval(self.get_input_string_port('use_lowest_val_loss', default='True'))
        self.test = ast.literal_eval(self.get_input_string_port('test', default='True'))
        self.test_size = int(self.get_input_string_port('test_size', default='1000'))
        self.lr_1 = float(self.get_input_string_port('learning_rate', default='0.001'))
        self.lr_2 = float(self.get_input_string_port('learning_rate_2', default='0.01'))
        self.max_pixel_intensity = float(self.get_input_string_port('max_pixel_intensity', default='255'))
        self.kernel_size = int(self.get_input_string_port('kernel_size', default='3'))
        self.small_model = ast.literal_eval(self.get_input_string_port('small_model', default='False'))

        # Get input data port, navigate to chip directory
        self.chip_dir = self.get_input_data_port('chips')

        # Format working directory
        os.chdir(self.chip_dir) # !!!!! Now in chip directory for remainder of task
        os.makedirs('models') # Create directory to save model after each epoch
        ref_geoj = [f for f in os.listdir('.') if f.endswith('ref.geojson')][0]
        chips = [f for f in os.listdir('.') if f.endswith('.tif')]

        # Get chip and reference geojson info
        self.geojson = os.path.join(self.chip_dir, ref_geoj)
        self.chips = [os.path.join(self.chip_dir, ch) for ch in chips]
        self.n_bands = gdal.Open(self.chips[0]).RasterCount
        logging.info('Detected number of bands: ' + str(self.n_bands))

        # Get input_shape info
        if self.resize_dim:
            self.input_shape = [self.n_bands, self.resize_dim, self.resize_dim]
        else:
            self.input_shape = [self.n_bands, self.max_side_dim, self.max_side_dim]
        logging.info('Input shape: ' + str(self.input_shape))

        # Create output data ports
        self.out_dir = self.get_output_data_port('trained_model')
        self.model_weights = os.path.join(self.out_dir, 'model_weights')
        self.rnd_1 = os.path.join(self.model_weights, 'round_1')
        self.info_dir = os.path.join(self.out_dir, 'info')
        os.makedirs(self.out_dir)
        os.makedirs(self.model_weights)
        os.makedirs(self.rnd_1)
        os.makedirs(self.info_dir)

        if self.two_rounds:
            self.rnd_2 = os.path.join(self.model_weights, 'round_2')
            os.makedirs(self.rnd_2)

        # Create class reference for one-hot encoded classes
        self.class_dict = {}
        for clss_ix in xrange(len(self.classes)):
            class_array = np.zeros(len(self.classes))
            class_array[clss_ix] = 1.
            self.class_dict[self.classes[clss_ix]] = class_array


    def filter_chips(self):
        '''
        Remove chips that are too large or too small. Remove entries in ref.geojson that
            are not valid chips.
        This method should be called from the chips directory, which contains the
            reference geojson for each chip.
        '''
        # Open reference geojson
        with open(self.geojson) as f:
            data = geojson.load(f)

        feature_collection = data['features']
        print str(len(feature_collection)) + ' initial features'
        valid_feats = []

        # Filter chips
        for feat in feature_collection:
            chip_name = str(feat['properties']['feature_id']) + '.tif'
            chip = gdal.Open(chip_name)

            try:
                min_side = min(chip.RasterXSize, chip.RasterYSize)
                max_side = max(chip.RasterXSize, chip.RasterYSize)
            except (AttributeError):
                logging.debug('Chip not found in directory: ' + chip_name)
                continue

            if min_side < self.min_side_dim or max_side > self.max_side_dim:
                logging.debug('Wrong sized chip: ' + chip_name)
            else:
                valid_feats.append(feat)

        data['features'] = valid_feats
        print str(len(valid_feats)) + ' features'

        # Replace self.geojson with only valid polygons
        with open('filtered.geojson', 'wb') as f:
            geojson.dump(data, f)

        return 'filtered.geojson'


    def format_geojson(self):
        '''
        Prep geojson by balancing (if necessary) and doing a train/test split.
            returns input geojson filepath.
        '''
        train_geoj = self.geojson

        # Create train/test data geojsons
        if self.test:
            gt.create_train_test(self.geojson, output_file='input.geojson',
                                 test_size=self.test_size)
            train_geoj, test_geoj = 'train_input.geojson', 'test_input.geojson'

        # Create balanced datasets
        if self.two_rounds:
            gt.create_balanced_geojson(train_geoj, classes=self.classes,
                                       output_file='train_balanced.geojson')
            train_geoj = 'train_balanced.geojson'

        # Confirm adequate amount of training data for train_size
        with open(train_geoj) as inp_file:
            poly_ct = len(geojson.load(inp_file)['features'])

        if poly_ct < self.train_size:
            raise Exception('There are only {} polygons that can be used as training ' \
                            'data, cannot train the network on {} samples. Please ' \
                            'decrease train_size or provide more ' \
                            'polygons.'.format(str(poly_ct), str(self.train_size)))

        # Return name of input training file
        return train_geoj


    def get_chips_from_features(self, feature_collection):
        '''
        Load chips into memory from a list of features
        Each chip will be padded to the input side dimension
        '''
        X, y, chip_names = [], [], []

        # Get chip names for each feature
        for feat in feature_collection:
            name = str(feat['properties']['feature_id']) + '.tif'
            clss = str(feat['properties']['class_name'])
            chip_names.append([name, clss])

        for chip, clss in chip_names:
            raster_array = []
            ds = gdal.Open(chip)
            bands = ds.RasterCount

            # Create normed raster array
            for n in xrange(1, bands + 1):
                raster_array.append(ds.GetRasterBand(n).ReadAsArray() / self.max_pixel_intensity)

            # pad to input shape
            chan, h, w = np.shape(raster_array)
            pad_h, pad_w = self.max_side_dim - h, self.max_side_dim - w
            chip_patch = np.pad(raster_array, [(0, 0), (pad_h/2, (pad_h - pad_h/2)),
                                (pad_w/2, (pad_w - pad_w/2))], 'constant',
                                constant_values=0)

            # resize chip
            if self.resize_dim:
                new_chip = []
                for band_ix in xrange(len(chip_patch)):
                    new_chip.append(imresize(chip_patch[band_ix],
                                    self.input_shape[-2:]).astype(float))
                chip_patch = np.array(new_chip)

            # Add raster array and one-hot class to X and y
            X.append(chip_patch)
            y.append(self.class_dict[clss])

            # sys.stdout.write('\r%{0:.2f}'.format(100 * (len(y)) / float(len(feature_collection))) + ' ' * 20)
            # sys.stdout.flush()

        return np.array(X), np.array(y)


    def compile_architecture(self):
        '''
        Implementation of VGG 16-layer net.
        '''
        print 'Compiling VGG Net...'

        model = Sequential()
        model.add(ZeroPadding2D((1,1), input_shape=self.input_shape))
        model.add(Convolution2D(64, self.kernel_size, self.kernel_size, activation='relu',
                                input_shape=self.input_shape))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(64, self.kernel_size, self.kernel_size,
                                activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, self.kernel_size, self.kernel_size,
                                activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(128, self.kernel_size, self.kernel_size,
                                activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, self.kernel_size, self.kernel_size,
                                activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, self.kernel_size, self.kernel_size,
                                activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(256, self.kernel_size, self.kernel_size,
                                activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, self.kernel_size, self.kernel_size,
                                activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, self.kernel_size, self.kernel_size,
                                activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, self.kernel_size, self.kernel_size,
                                activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, self.kernel_size, self.kernel_size,
                                activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, self.kernel_size, self.kernel_size,
                                activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, self.kernel_size, self.kernel_size,
                                activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(Flatten())
        model.add(Dense(2048, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2048, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(len(self.classes), activation='softmax'))

        sgd = SGD(lr=self.lr_1, decay=0.01, momentum=0.9, nesterov=True)
        model.compile(optimizer = 'sgd', loss = 'categorical_crossentropy')
        return model


    def train_model(self, model, train_geojson, retrain=False):
        '''
        Train model using the chips referenced in train_geojson. Returns a trained
            model and a history of all validation losses
        '''
        validation_data, full_hist = None, []
        rnd = '2' if retrain else '1'

        with open(train_geojson) as f:
            feats = geojson.load(f)['features']
            np.random.shuffle(feats)

        # Freeze layers if retraining model
        nb_epoch = self.nb_epoch if not retrain else self.nb_epoch_2
        train_size = self.train_size if not retrain else self.train_size_2
        if retrain:
            for i in xrange(len(model.layers[:-1])):
                model.layers[i].trainable = False
                sgd = SGD(lr=self.lr_2, momentum=0.9, nesterov=True)
                model.compile(loss='categorical_crossentropy', optimizer='sgd')

        # Get validation data (10% of training data)
        val_size = int(0.1 * self.train_size)
        val_data, train_data = feats[:val_size], feats[val_size: train_size]
        train_size = len(train_data)
        valX, valY = self.get_chips_from_features(val_data)

        for e in range(nb_epoch):
            print 'Epoch {}/{}'.format(e + 1, nb_epoch)
            chk = ModelCheckpoint(filepath="./models/epoch" + str(e) + \
                                  "_{val_loss:.2f}.h5", verbose=1)
            np.random.shuffle(train_data)

            # Cycle through batches of chips and train
            for batch_start in range(0, train_size, 1000):
                callbacks = []
                this_batch = train_data[batch_start: batch_start + 1000]

                # Generate chips
                X,y = self.get_chips_from_features(this_batch)
                if batch_start == range(0, train_size, 1000)[-1]:
                    callbacks = [chk]

                # Fit the model
                hist = model.fit(X, y, batch_size=self.batch_size, nb_epoch=1,
                                 validation_data=(valX, valY), callbacks=callbacks)

            full_hist.append(hist.history)

        # Load model weights with the lowest validation loss
        if self.use_lowest_val_loss and rnd == '1':
            val_losses = [epoch['val_loss'][0] for epoch in full_hist]
            min_epoch = np.argmin(val_losses)
            min_loss = val_losses[np.argmin(val_losses)]
            min_weights = 'models/epoch' + str(min_epoch) + '_{0:.2f}.h5'.format(min_loss)
            model.load_weights(min_weights)

        # Save all models to output dir
        save_path = os.path.join(self.model_weights, 'round_{}'.format(rnd))
        for weights in os.listdir('models/'):
            shutil.move('models/' + weights, save_path)

        return model


    def test_net(self, model):
        '''
        test network performance on test data
        '''
        y_pred, y_true, test_report = [], [], ''
        with open('test_input.geojson') as f:
            test_data = geojson.load(f)['features']

        for feat_ix in xrange(0, self.test_size, 1000):
            x, y = self.get_chips_from_features(test_data[feat_ix: feat_ix + 1000])

            y_pred += list(model.predict_classes(x))
            y_true += [int(np.argwhere(clss == 1)) for clss in y]

        test_size = len(y_true)
        y_pred, y_true = np.array(y_pred), np.array(y_true)

        test_report = classification_report(y_true, y_pred, target_names=self.classes)

        # Record test results
        with open(os.path.join(self.out_dir, 'test_report.txt'), 'w') as tr:
            tr.write(test_report)


    def invoke(self):
        '''
        Execute task
        '''
        # Filter chips and update reference geojson
        self.geojson = self.filter_chips()

        # Get train, test and balanced geojsons
        train_geojson = self.format_geojson()

        # Create model architecture
        model = self.compile_architecture()

        # Fit model rnd 1
        mem_error = 'Model does not fit in memory. Please try one or more of the ' \
                    'following: (1) Use resize_dim to downsample chips. (2) Use a ' \
                    'smaller batch_size. (3) Set the small_model flag to True.'
        try:
            model = self.train_model(model=model, train_geojson=train_geojson)
        except (MemoryError):
            raise Exception(mem_error)

        # Training round two
        if self.two_rounds:
            try:
                model = self.train_model(model=model, retrain=True,
                                         train_geojson='train_input.geojson')
            except (MemoryError):
                raise Exception(mem_error)


        if self.test:
            self.test_net(model=model)

        # Save model and classes to output directory
        model.save(os.path.join(self.out_dir, 'trained_model.h5'))

        classes_json = {'classes': str(self.classes)}
        with open(os.path.join(self.info_dir, 'classes.json'), 'wb') as f:
            json.dump(classes_json, f)

        # Save train and test data to info directory of output
        shutil.copy(train_geojson, os.path.join(self.info_dir, 'train_data.geojson'))
        if self.test:
            shutil.copy('test_input.geojson', os.path.join(self.info_dir, 'test_data.geojson'))

if __name__ == '__main__':
    with TrainCnnChipClassifier() as task:
        task.invoke()
