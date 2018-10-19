import pandas as pd
import numpy as np
#  import matplotlib.pyplot as plt
import pylab
import os
import pydicom
import random
import matplotlib.pyplot as plt
from glob import glob
from skimage.transform import resize
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from PIL import Image
from keras.applications.densenet import DenseNet121 as PTModel, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization, GlobalAveragePooling2D, Dense, Dropout, Input
# , AvgPool2D, Lambda, LocallyConnected2D, Conv2D, multiply, Flatten
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau\
    # , LearningRateScheduler
from keras.utils import Sequence
import keras_preprocessing.image as KPImage

# params we will probably want to do some hyperparameter optimization later

# ['InceptionV3', 'Xception', 'DenseNet169', 'VGG16']
BASE_MODEL = 'DenseNet121'
IMG_SIZE = (384, 384)  # [(224, 224), (384, 384), (512, 512), (640, 640)]
BATCH_SIZE = 24  # [1, 8, 16, 24]
DENSE_COUNT = 128  # [32, 64, 128, 256]
DROPOUT = 0.25  # [0, 0.25, 0.5]
LEARN_RATE = 1e-4  # [1e-4, 1e-3, 4e-3]
TRAIN_SAMPLES = 8000  # [3000, 6000, 15000]
TEST_SAMPLES = 800
USE_ATTN = False  # [True, False]

# image_bbox_df = pd.read_csv('../input/lung-opacity-overview/image_bbox_full.csv')
# image_bbox_df['path'] = image_bbox_df['path'].map(lambda x:
#                              x.replace('input',
#                                        'input/rsna-pneumonia-detection-challenge'))


# Labels contains the target (1=pneumonia, 0=healthy) and bounding boxes
# if that patient has pneumonia
bbox_df = pd.read_csv('../input/stage_1_train_labels.csv')

# Detailed contains whether patient has lung opacity, image is not normal or
# they are healthy
det_class_df = pd.read_csv('../input/stage_1_detailed_class_info.csv')
det_class_df = det_class_df.groupby('patientId').head(1).reset_index()
# Join the two tables
comb_bbox_df = pd.merge(bbox_df, det_class_df, how='outer', on='patientId')
comb_bbox_df.head(3)

# Initialize paths to image directories
det_class_path = '../input/stage_1_detailed_class_info.csv'
bbox_path = '../input/stage_1_train_labels.csv'
dicom_dir = '../input/stage_1_train_images/'
path = '../input/stage_1_train_images/%s.dcm' % comb_bbox_df['patientId'][0]
dicom_header = pydicom.read_file(path, stop_before_pixels=True)


def get_header_info(patientId):
    """Function to apply to dataframe to insert all header info from DICOM
    image as separate columns

    :param patientId str: Unique ID
    :returns: A Pandas Series that will become a column
    """
    path = '../input/stage_1_train_images/%s.dcm' % patientId
    output = {'path': path}
    dicom_header = pydicom.read_file(path, stop_before_pixels=True)
    for value in dicom_header:
        output[value.name] = value.value
    return pd.Series(output)


comb_bbox_df.columns.values
header_df = comb_bbox_df.apply(lambda x: get_header_info(x['patientId']), 1)
header_df['Patient\'s Age'] = header_df['Patient\'s Age'].map(int)
header_df['Patient\'s Age'].hist()
header_df.columns.values
header_df.shape[0]

header_df = header_df.groupby('Patient ID').head(1).reset_index()
# This contains all information from the header and from the label file
image_full_df = pd.merge(header_df, comb_bbox_df,
                         left_on='Patient ID', right_on='patientId')
# Columns are:
#   ['index_x', 'path', 'Specific Character Set', 'SOP Class UID',
#  'SOP Instance UID', 'Study Date', 'Study Time', 'Accession Number',
#  'Modality', 'Conversion Type', "Referring Physician's Name",
#  'Series Description', "Patient's Name", 'Patient ID',
#  "Patient's Birth Date", "Patient's Sex", "Patient's Age",
#  'Body Part Examined', 'View Position', 'Study Instance UID',
#  'Series Instance UID', 'Study ID', 'Series Number',
#  'Instance Number', 'Patient Orientation', 'Samples per Pixel',
#  'Photometric Interpretation', 'Rows', 'Columns', 'Pixel Spacing',
#  'Bits Allocated', 'Bits Stored', 'High Bit',
#  'Pixel Representation', 'Lossy Image Compression',
#  'Lossy Image Compression Method', 'patientId', 'x', 'y', 'width',
#  'height', 'Target', 'index_y', 'class']

#  Uncomment to show an example image
#  patientId = image_full_df['patientId'][0]
#  dcm_file = '../input/stage_1_train_images/%s.dcm' % patientId
#  dcm_data = pydicom.read_file(dcm_file)
#  im = dcm_data.pixel_array  # Numpy array containing uint8s, 1024 x 1024
#  pylab.imshow(im, cmap=pylab.cm.gist_gray)
#  pylab.axis('off')


# Predict Pneumonia
class_enc = LabelEncoder()
image_full_df['class_idx'] = class_enc.fit_transform(image_full_df['class'])
encoder = OneHotEncoder(sparse=False)
image_full_df['class_vec'] = encoder.fit_transform(
    image_full_df['class_idx'].values.reshape(-1, 1)).tolist()


def group_bbox(df):
    """Group x, y, w, h together for each image

    :df: Pandas dataframe
    :returns: Pandas series
    """
    return pd.Series({'patientId': df['patientId'].sample(1)
                      , 'bbox': [df['x'], df['y'], df['width'], df['height']]
                      , 'class': df['class'].sample(1)
                      })


grouped_bbox_df = comb_bbox_df.groupby('patientId').apply(group_bbox)
grouped_bbox_df.set_index('patientId')

raw_train_df, valid_df = train_test_split(grouped_bbox_df, test_size=0.25, random_state=2018,
                                          stratify=grouped_bbox_df['class'])


class generator(Sequence):
    def __init__(self,
                 df,
                 patientIds,
                 batch_size=32,
                 image_size=256,
                 do_shuffle=True,
                 augment=False,
                 predict=False
                 ):
        self.df = df
        self.ids = patientIds
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = do_shuffle
        self.augment = augment
        self.predict = predict
        self.on_epoch_end()

    def __load__(self, pid):
        # load dicom file as numpy array
        img = pydicom.dcmread(os.path.join(self.folder, pid, ".dcm")).pixel_array
        # create empty mask
        msk = np.zeros(img.shape)
        # if image contains pneumonia
        boxes = self.df.loc[self.df['patientId'] == pid]['bbox']
        if boxes:
            # loop through pneumonia
            for box in boxes:
                # add 1's at the location of the pneumonia
                x, y, w, h = box
                msk[y:y + h, x:x + w] = 1
        # resize both image and mask
        img = resize(img, (self.image_size, self.image_size), mode='reflect')
        msk = resize(msk, (self.image_size, self.image_size), mode='reflect') > 0.5
        # if augment then horizontal flip half the time
        if self.augment and random.random() > 0.5:
            img = np.fliplr(img)
            msk = np.fliplr(msk)
        # add trailing channel dimension
        img = np.expand_dims(img, -1)
        msk = np.expand_dims(msk, -1)
        return img, msk

    def __loadpredict__(self, pid):
        # load dicom file as numpy array
        img = pydicom.dcmread(os.path.join(self.folder, pid, ".dcm")).pixel_array
        # resize image
        img = resize(img, (self.image_size, self.image_size), mode='reflect')
        # add trailing channel dimension
        img = np.expand_dims(img, -1)
        return img

    def __getitem__(self, index):
        # select batch
        filenames = self.filenames[index * self.batch_size:(index + 2) * self.batch_size]
        # predict mode: return images and filenames
        if self.predict:
            # load files
            imgs = [self.__loadpredict__(patientId) for patientId in self.ids]
            # create numpy batch
            imgs = np.array(imgs)
            return imgs, filenames
        # train mode: return images and masks
        else:
            # load files
            items = [self.__load__(patientId) for patientId in self.ids]
            # unzip images and masks
            imgs, msks = zip(*items)
            # create numpy batch
            imgs = np.array(imgs)
            msks = np.array(msks)
            return imgs, msks

    def on_epoch_end(self):
        #  if self.shuffle:
        #      random.shuffle(self.filenames)
        self.df = shuffle(self.ids)  # sklearn shuffle

    def __len__(self):
        if self.predict:
            # return everything
            return int(np.ceil(len(self.ids) / self.batch_size))
        else:
            # return full batches only
            return int(len(self.df) / self.batch_size)


skf = StratifiedKFold(n_splits=2)
train_ids = []
valid_ids = []
pid_vec = np.stack(grouped_bbox_df['patientId'].values)[:, 0]
target_vec = np.stack(grouped_bbox_df['target'].values)[:, 0]
for t_ids, v_ids in skf.split(pid_vec, target_vec):
    train_ids = pid_vec[t_ids]
    valid_ids = target_vec[v_ids]
    break

folder = '../input/stage_1_train_images'
train_gen = generator(folder,
                      bbox_df,
                      train_ids,
                      batch_size=32,
                      image_size=256,
                      do_shuffle=True,
                      augment=True,
                      predict=False
                      )
valid_gen = generator(folder,
                      bbox_df,
                      valid_ids,
                      batch_size=32,
                      image_size=256,
                      do_shuffle=True,
                      augment=True,
                      predict=False
                      )

train_src, train_tar = next(train_gen)
valid_src, valid_tar = next(valid_gen)

print(train_src.shape, train_tar.shape)
fig, m_axs = plt.subplots(2, 4, figsize=(16, 8))
for (c_x, c_y, c_ax) in zip(train_src, train_tar, m_axs.flatten()):
    c_ax.imshow(c_x[:, :, 0], cmap='bone')
    c_ax.set_title('%s' % class_enc.classes_[np.argmax(c_y)])
    c_ax.axis('off')
