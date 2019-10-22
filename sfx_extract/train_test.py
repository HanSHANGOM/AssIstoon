#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Mask R-CNN
Train on the nuclei segmentation dataset from the
Kaggle 2018 Data Science Bowl
https://www.kaggle.com/c/data-science-bowl-2018/

Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from ImageNet weights
    python3 Manga.py train --dataset=/path/to/dataset --subset=train --weights=imagenet

    # Train a new model starting from specific weights file
    python3 Manga.py train --dataset=/path/to/dataset --subset=train --weights=/path/to/weights.h5

    # Resume training a model that you had trained earlier
    python3 Manga.py train --dataset=/path/to/dataset --subset=train --weights=last

    # Generate submission file
    python3 Manga.py detect --dataset=/path/to/dataset --subset=train --weights=<last or /path/to/weights.h5>
"""

# Set matplotlib backend
# This has to be done before other importa that might
# set it, but only if we're running in script mode
# rather than being imported.
if __name__ == '__main__':
    import matplotlib
    # Agg backend runs without a display
    matplotlib.use('Agg')

    import matplotlib.pyplot as plt

import os
import sys
import json
import datetime
import numpy as np
import skimage.io
from imgaug import augmenters as iaa


# In[2]:




# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize

# Path to trained weights file
#COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
COCO_WEIGHTS_PATH= os.path.join('D:/logs/manga20190811T2349', "mask_rcnn_manga_1139.h5")
# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = "D:/logs/"
#DEFAULT_LOGS_DIR = "C:\\Users\\sanghoon\\AssIstoon\\sfx_extract\\dataset\\logs\\"
#os.path.join(ROOT_DIR, "logs")

# Results directory
# Save submission files here
RESULTS_DIR = os.path.join(ROOT_DIR, "results/Manga/")

# The dataset doesn't have a standard train/val split, so I picked
# a variety of images to surve as a validation set.

VAL_IMAGE_IDS = [name.split('.')[0] for name in os.listdir('C:/Users/sanghoon/AssIstoon/sfx_extract/dataset/val/images/')]



# In[10]:


############################################################
#  Configurations
############################################################

class MangaConfig(Config):
    """Configuration for training on the Manga segmentation dataset."""
    # Give the configuration a recognizable name
    NAME = "Manga"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Adjust depending on your GPU memory
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + Manga

    # Number of training and validation steps per epoch
    STEPS_PER_EPOCH = 10

    VALIDATION_STEPS = 10
    MINI_MASK_SHAPE = (256, 256)
    IMAGE_RESIZE_MODE = "none"


    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between Manga and BG
    #DETECTION_MIN_CONFIDENCE = 0.9


class MangaInferenceConfig(MangaConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7


# In[18]:



############################################################
#  Dataset
############################################################
import random
class MangaDataset(utils.Dataset):

    def load_Manga(self,
                   dataset_dir,
                   subset):
        """망가데이터셋의 서브셋 불러오기.

        dataset_dir: 데이터셋의 루트 디렉토리
        subset: 불러올 서브셋. 서브디렉토리의 이름,
                예를 들면 stage1_train, stage1_test, ...etc. 또는, 다음 중 하나:
                * train: stage1_train excluding validation images
                * val: VAL_IMAGE_IDS에 해당하는 validation 이미지
        """
        # 클래스를 추가한다
        # Naming the dataset Manga, and the class Manga
        self.add_class("Manga", 1, "sfx")
        #self.add_class("Manga", 2, "blank")

        # 어떤 subset? val or train?
        # "val": use hard-coded list above
        # "train": use data from stage1_train minus the hard-coded list above
        # else: use the data from the specified sub-directory
        #assert subset in ["train", "val", "stage1_train", "stage1_test", "stage2_test"]
        subset_dir = "train" if subset in ["train", "val"] else subset #수정필요
        #dataset_dir = 'C:/Users/sanghoon/AssIstoon/sfx_extract/dataset/'
        root_dir= dataset_dir+subset+'/'
        images_dir = root_dir+'images/'
        if subset=="train" :
            image_ids = [name[:-4] for name in os.listdir(images_dir)]  # 디렉토리 이름에서 ? 이미지 아이디를 받는다.
            #random.shuffle(image_ids)
        elif subset=="val":
            image_ids=VAL_IMAGE_IDS
        # Add images
        for image_id in image_ids:
            self.add_image(
                source="Manga",
                image_id=image_id,
                path=images_dir+'{}.jpg'.format(image_id),
                root=root_dir,
                subset=subset
            )
                
    def load_mask(self, image_id):
        """이미지에 대한 instance mask를 generate한다.
       반환값:
        masks: 형상의 bool array [height, width, instance count] 
            인스턴스마다 한 마스크로.
        class_ids: 인스턴스 마스크의 클래스 아이디로 이루어진 1D array
        """
        info = self.image_info[image_id] #image id에 대한 info를 얻어온다.

        # 이미지 경로로부터 마스크 디렉토리를 얻어온다.
        mask_dir = os.path.join(info['root'], "masks")
        # png이미지로부터 mask파일을 읽는다.
        mask_list = [f for f in os.listdir(mask_dir) if
                     f.split('_')[0] == info['id'].split('_')[0] + info['id'].split('_')[1] and f.endswith(".png")]
        print(info['id'],mask_list)
        mask=[]
        for f in mask_list:

            m = skimage.io.imread(os.path.join(mask_dir, f))
            z=np.zeros(m.shape[:2])
            for h in range(z.shape[0]):
                for w in range(z.shape[1]):
                    z[h,w]= np.sum(m[h,w,:])
            m=z.astype(np.bool)

            #mask[m==True]=1
            mask.append(m)
        #mask=mask.astype(np.bool)
        #print(mask.shape)
        mask = np.stack(mask, axis=-1)


        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        return mask, np.ones(mask.shape[-1], dtype=np.int32)

    def image_reference(self, image_id):
        """이미지의 경로를 반환한다."""
        info = self.image_info[image_id]
        if info["source"] == "Manga":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)


# In[19]:



############################################################
#  Training
############################################################

def train(model, dataset_dir):
    """Train the model."""
    # Training dataset.
    dataset_train = MangaDataset()
    dataset_train.load_Manga(dataset_dir, "train")
    dataset_train.prepare() #dataset 클래스의 참조 필요


    # Validation dataset
    dataset_val = MangaDataset()
    dataset_val.load_Manga(dataset_dir, "val")
    dataset_val.prepare()

    # Image augmentation
    augmentation = iaa.SomeOf((2, 4), [
    #iaa.Fliplr(0.5),
    #iaa.Flipud(0.5),
    iaa.OneOf([iaa.Affine(rotate=15),
               iaa.Affine(rotate=10),
               iaa.Affine(rotate=20),
               iaa.Affine(rotate=25),
               iaa.Affine(rotate=30),
               iaa.Affine(rotate=350),
               iaa.Affine(rotate=345),
               iaa.Affine(rotate=340),
               iaa.Affine(rotate=330)]),
    #iaa.Multiply((0.8, 1.5)),
    #iaa.GaussianBlur(sigma=(0.0, 5.0)),
    iaa.Dropout(p=(0.15,0.25)),
    iaa.Pepper(p=(0.2,0.3)),
    iaa.CoarseDropout(p=(0.2,0.6),size_percent=(0.02, 0.4)),
    iaa.AdditiveGaussianNoise(scale=0.05*255)])
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    #이건 뭐하는걸까? image augmentation인걸 보니 전처리를 해주는가보다.

    # *** 수정이 필요하면 맞게 수정하세요 ***

    # If starting from imagenet, train heads only for a bit
    # since they have random weights
    print("Train network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE, #config 참조 필요 혹은 직접 러닝레이트 지정
                epochs=150,
                augmentation=augmentation,
                layers='heads')

    print("Train all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=2000,
                augmentation=augmentation,
                layers='all')


# In[20]:



############################################################
#  RLE Encoding
############################################################
#RLE란 무엇인가? 간단한 비손실 압축 방법으로 데이터에서 같은 값이 연속해서 나타나는 것을 그 개수와 반복되는 값만으로 표시하는 방법.
def rle_encode(mask): #mask를 RLE 인코딩 하나보다
    """Encodes a mask in Run Length Encoding (RLE).
    Returns a string of space-separated values.
    """
    assert mask.ndim == 2, "mask는 다음과 같은 모양이여야 한다. [Height, Width]"
    # Flatten it column wise 
    m = mask.T.flatten()
    # Compute gradient. Equals 1 or -1 at transition points
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # 1-based indicies of transition points (where gradient != 0)
    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
    # Convert second index in each pair to lenth
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    return " ".join(map(str, rle.flatten()))


def rle_decode(rle, shape): #압축한 mask 인코딩 데이터를 디코딩한다.
    """Decodes an RLE encoded list of space separated
    numbers and returns a binary mask."""
    rle = list(map(int, rle.split()))
    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
    rle[:, 1] += rle[:, 0]
    rle -= 1
    mask = np.zeros([shape[0] * shape[1]], np.bool)
    for s, e in rle:
        assert 0 <= s < mask.shape[0]
        assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)
        mask[s:e] = 1
    # Reshape and transpose
    mask = mask.reshape([shape[1], shape[0]]).T
    return mask


def mask_to_rle(image_id, mask, scores): #
    "Encodes instance masks to submission format."
    assert mask.ndim == 3, "Mask must be [H, W, count]"
    # If mask is empty, return line with image ID only
    if mask.shape[-1] == 0:
        return "{},".format(image_id)
    # 겹치는 mask 제거
    # Multiply each instance mask by its score order
    # then take the maximum across the last dimension #점수 제일 높은걸 고른다는 듯.
    order = np.argsort(scores)[::-1] + 1  # 1-based descending
    mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
    # Loop over instance masks
    lines = []
    for o in order:
        m = np.where(mask == o, 1, 0)
        # Skip if empty
        if m.sum() == 0.0:
            continue
        rle = rle_encode(m)
        lines.append("{}, {}".format(image_id, rle))
    return "\n".join(lines)


# In[21]:



############################################################
#  Detection
############################################################

def detect(model, dataset_dir, subset):
    """Run detection on images in the given directory."""
    print("Running on {}".format(dataset_dir))

    # Create directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(RESULTS_DIR, submit_dir)
    os.makedirs(submit_dir)

    # Read dataset
    dataset = MangaDataset()
    dataset.load_Manga(dataset_dir, subset)
    dataset.prepare()
    # Load over images
    submission = []
    for image_id in dataset.image_ids:
        # Load image and run detection
        image = dataset.load_image(image_id)
        # Detect objects
        r = model.detect([image], verbose=0)[0]
        # Encode image to RLE. Returns a string of multiple lines
        source_id = dataset.image_info[image_id]["id"]
        rle = mask_to_rle(source_id, r["masks"], r["scores"])
        submission.append(rle)
        # Save image with masks
        visualize.display_instances(
            image, r['rois'], r['masks'], r['class_ids'],
            dataset.class_names, r['scores'],
            show_bbox=False, show_mask=False,
            title="Predictions")
        plt.savefig("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"]))

    # Save to csv file
    submission = "ImageId,EncodedPixels\n" + "\n".join(submission)
    file_path = os.path.join(submit_dir, "submit.csv")
    with open(file_path, "w") as f:
        f.write(submission)
    print("Saved to ", submit_dir)


# In[24]:


############################################################
#  Command Line
############################################################

if __name__ == '__main__':
    import argparse
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # Parse command line arguments
    '''
    parser = argparse.ArgumentParser(
        description='Mask R-CNN for AssIstoon counting and segmentation')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'detect'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dataset/",
                        help='Root directory of the dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5", #수정 필요? 
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--subset', required=False,
                        metavar="Dataset sub-directory",
                        help="Subset of dataset to run prediction on")
    args = parser.parse_args()

    '''

    '''dataset_train = MangaDataset()
    dataset_train.load_Manga()
    dataset_train.prepare()
    print("Image Count: {}".format(len(dataset_train.image_ids)))
    print("Class Count: {}".format(dataset_train.num_classes))
    for i, info in enumerate(dataset_train.class_info):
        print("{:3}. {:50}".format(i, info['name']))

    image_ids = np.random.choice(dataset_train.image_ids, 4)
    print(image_ids)

    for image_id in image_ids:

        image = dataset_train.load_image(image_id)
        mask, class_ids = dataset_train.load_mask(image_id)
        visualize.display_top_masks(image, mask, class_ids,dataset_train.class_names, limit=1)
'''
    command="train"

    if command=='train':
        config = MangaConfig()
    else:
        config = MangaInferenceConfig()
    config.display()

    # Create model
    #if args.command == "train":
    if command=="train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=DEFAULT_LOGS_DIR)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=DEFAULT_LOGS_DIR)
    weights="coco"
    # Select weights file to load
    #if args.weights.lower() == "coco":
    if weights=="coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = None

    # Load weights
    print("Loading weights ", weights_path)
    if weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True
                           #, exclude=[
            #"mrcnn_class_logits",
            #"mrcnn_bbox_fc",
            #"mrcnn_bbox",
            #"mrcnn_mask"
            #]
        )

    else:
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits",
            "mrcnn_bbox_fc",
            "mrcnn_bbox",
            "mrcnn_mask"
            ])

    dataset = "C:/Users/sanghoon/AssIstoon/sfx_extract/dataset/"
    if command == "train":
        train(model, dataset)
    elif command == "detect":
        detect(model, dataset, subset="train")
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(command))

    # Train or evaluate


# In[ ]:





# In[ ]:




