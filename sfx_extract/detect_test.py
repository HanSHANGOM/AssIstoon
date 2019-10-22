"""
Mask R-CNN
Base Configurations class.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
import sys
import numpy as np
import skimage.io
import cv2 as cv
ROOT_DIR = os.path.abspath("../")
print(ROOT_DIR)
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib

class MangaConfig(Config):  
    NAME = "Manga"
    IMAGE_RESIZE_MODE = 'pad64'
    IMAGE_MIN_DIM = 768
    IMAGE_MAX_DIM = 1216
    GPU_COUNT = 1
    
    IMAGES_PER_GPU = 2

    LEARNING_RATE = 0.0001
    TRAIN_BN = False
    BACKBONE = "resnet50"
    NUM_CLASSES = 1 + 3  

    USE_MINI_MASK = False
    MINI_MASK_SHAPE = (112, 112)


    STEPS_PER_EPOCH = 8
    VALIDATION_STEPS = 5

    MAX_GT_INSTANCES = 30

    DETECTION_MIN_CONFIDENCE = 0.9
    RPN_NMS_THRESHOLD = 0.9

############################################################
#  Dataset
############################################################

class MangaDataset(utils.Dataset):  # 데이터셋이야.
    def __init__(self):
        super(MangaDataset, self).__init__()
        """
        가로세로를 최대크기에 맞게 64의 배수로 조정하기 위해 사이즈에 대해 값을 저장할 변수들임.
        """
        self.maxh = -1
        self.maxw = -1

    def load_Manga(self,
                   data_dir,
                   subset=None, config=None):
        """망가데이터셋의 서브셋 불러오기.
        data_dir: 데이터셋의 루트 디렉토리
    ​
        """
        # 클래스를 추가한다
        # Naming the dataset Manga, and the class Manga
        self.add_class("Manga", 1, "sfx")
        self.add_class("Manga", 2, "text")
        self.add_class("Manga", 3, "face")


        image_ids = [name[:-4] for name in os.listdir(data_dir)]  # 디렉토리 이름에서 ? 이미지 아이디를 받는다.
        # print(type(self.image_info))
        for idx, image_id in enumerate(image_ids):  # 일단 이미지를 그대로 다 받는다.
            try:
                image_array = skimage.io.imread(os.path.join(data_dir, image_id + '.jpg'))  # 이미지를 따오고
            except:
                image_array = skimage.io.imread(os.path.join(data_dir, image_id + '.png'))
            if image_array.ndim != 3:  # RGB로 변환
                image_array = skimage.color.gray2rgb(image_array)
            self.add_image(source="Manga", image_id=image_id, path=data_dir + '{}.jpg'.format(image_id), index=idx,
                           shape=image_array.shape, image=image_array,
                           window=None)  # 이미지에 대한 정보 저장. image_id는 확장자 제외 이미지 이름, shape는 padding 전 original size, window...도 비슷한 느낌으로 저장됨.
            h, w = self.image_info[idx]['shape'][:2]  # 여러개일 경우를 상정해서 maxh,maxw를 저장하는 부분
            if self.maxh < h:
                self.maxh = h
            if self.maxw < w:
                self.maxw = w
            print(self.maxh,self.maxw)
        for idx, image_id in enumerate(image_ids):  # 최대 사이즈가 정해졌으니 padding.
            self.pad64(idx)
            print(self.maxh,self.maxw)

    def pad64(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = self.image_info[image_id]['image']  # 저장한 정보에서 이미지를 받는다.



        h, w = self.image_info[image_id]['shape'][:2]  # 오리지널 사이즈를 받는다.

        """
        아래 코드는 패딩할 사이즈를 계산
        """

        if self.maxh % 64 > 0:
            self.maxh = self.maxh - (self.maxh % 64) + 64
            top_pad = (self.maxh - h) // 2
            bottom_pad = self.maxh - h - top_pad
        else:
            top_pad = bottom_pad = 0
        # Width
        if self.maxw % 64 > 0:
            self.maxw = self.maxw - (self.maxw % 64) + 64
            left_pad = (self.maxw - w) // 2
            right_pad = self.maxw - w - left_pad
        else:
            left_pad = right_pad = 0
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=255)  # 주위를 다 흰색으로 채운다.
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
        # print(image_id,image.shape)
        self.image_info[image_id]['image'] = image
        self.image_info[image_id]['window'] = window
        """
         padding한 이미지로 변환해서 image array를 데이터셋에 저장.
        """

        # from PIL import Image
        # img = Image.fromarray(image, 'RGB')
        # img.save('C:/Users/sanghoon/AssIstoon/pad64image.jpg')
        """
        테스트용
        """

    def load_mask(self, image_id):  # 학습용이므로 넘어가도 좋음.

        info = self.image_info[image_id]  # image id에 대한 info를 얻어온다.


        # 이미지 경로로부터 마스크 디렉토리를 얻어온다.
        mask_dir = os.path.join(info['root'], "masks")
        # png이미지로부터 mask파일을 읽는다.
        mask_list = [f for f in os.listdir(mask_dir) if
                     f.split('_')[0] == info['id'].split('_')[0] + info['id'].split('_')[1] and f.endswith(".png")]

        mask = []
        maskclasslist = []
        for f in mask_list:
            maskclass = f.split('_')[1].split('-')[0]
            if maskclass == 'sfx':
                maskclasslist.append(1)
            elif maskclass == 'text':
                maskclasslist.append(2)
            elif maskclass == "face":
                maskclasslist.append(3)
            m = skimage.io.imread(os.path.join(mask_dir, f))
            m = m.sum(axis=2)
            mask.append(m)

        mask = np.stack(mask, axis=-1)
        return mask, np.array(maskclasslist, dtype=np.int32)


    def image_reference(self, image_id):
        """이미지의 경로를 반환한다."""
        info = self.image_info[image_id]
        if info["source"] == "Manga":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)


class MangaInferenceConfig(MangaConfig):  # detect에 대한 config 부분임.
    IMAGE_RESIZE_MODE = "none"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    DETECTION_MIN_CONFIDENCE = 0.9
    RPN_NMS_THRESHOLD = 0.8

    def __init__(self, dataset):  #
        self.h = dataset.maxh
        self.w = dataset.maxw

        self.IMAGE_CHANNEL_COUNT = 3
        """Set values of computed attributes."""
        # Effective batch size
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT
        self.IMAGE_SHAPE = np.array([self.h, self.w,
                                     self.IMAGE_CHANNEL_COUNT])
        self.IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES

# detect를 도와주는 함수들.
def rle_encode(mask):
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


def rle_decode(rle, shape):  # 압축한 mask 인코딩 데이터를 디코딩한다.
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



def mask_to_rle(image_id, mask, scores):  #
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





# 각 마스크에 대해 빨간색깔을 칠해 적용해서 RGB 3차원 array로 변환하는 곳
def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    color = [255, 0, 0]
    result = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.int32)


    tmp = result[:, :, 0].copy()

    tmp[mask] = 255
    # print(tmp)
    result[:, :, 0] = tmp

    return result

def display_instances(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None, original_shape=None):
    # 이미지 하나에 대해서 마스크를 합친다.
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")  # 검출된 것이 없을경우.
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]
    image = image.astype(np.uint32).copy()
    padded_shape = masks[:, :,
                   0].shape  # masks는 2차원 array로 된 마스크 array들을 3차원 array로 담고 있는 것. 그래서 result의 shape를 따기 위해 잠시 그 첫번째를 활용.
    mask_list = []
    bbox_list = []
    pad_top = (padded_shape[0] - original_shape[0]) // 2
    pad_bottom = padded_shape[0] - original_shape[0] - pad_top
    pad_left = (padded_shape[1] - original_shape[1]) // 2
    pad_right = padded_shape[1] - original_shape[1] - pad_left
    # np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.int32)
    for i in range(N):
        if class_ids[i] != 1:  # 얼굴은 제외하고 칠한다는 뜻
            continue
        color = (255, 0, 0)
        # Mask
        mask = masks[:, :, i]


        masked_image = apply_mask(image, mask, color)
        masked_image = masked_image[pad_top:-pad_bottom, pad_left:-pad_right, :]
        #print(boxes[i])
        y1, x1, y2, x2 = boxes[i]
        bbox = y1 - pad_top, x1 - pad_left, y2 - pad_top, x2 - pad_left
        """
        이 부분은 마찬가지로 저장용
        """
        # r, g, b = cv.split(masked_image)
        # save=cv.merge([b,g,r])
        # path='./'
        # cv.imwrite(path+'masked_image{}.jpg'.format(i), save)
        mask_list.append(masked_image)
        bbox_list.append(bbox)

        """
        패딩한 사이즈를 다시 오리지널 사이즈로 crop하는 과정.
        """

    image = image[pad_top:-pad_bottom, pad_left:-pad_right, :]


    return image, mask_list, bbox_list  # 혹시 몰라서 masks도 반환함.


def detect(model, dataset):
    submission = []
    result = {}
    id_list = []
    for image_dict in dataset.image_info:

        image = image_dict['image']
        r = model.detect([image], verbose=0)[0]
        source_id = image_dict["id"]
        id_list.append(source_id)
        rle = mask_to_rle(source_id, r["masks"], r["scores"])
        submission.append(rle)
        image, mask_list, bbox_list = display_instances(
            image, r['rois'], r['masks'], r['class_ids'],
            dataset.class_names, r['scores'],
            show_bbox=False, show_mask=True,
            title=None, original_shape=image_dict['shape'])

    return image, mask_list, bbox_list



def extract_sfx(imagepath=None):
    import shutil
    # Path to trained weights file
    dataset_dir = './temp/'
    filename = imagepath.split('/')[-1]
    shutil.copy(imagepath, dataset_dir + filename)
    # 이미지 받을 폴더명. 한개씩이니까 임시폴더 지정이라고 생각해줘. 어디서 이미지를 받아서 임시로 여기 copy해놓고 이 이미지를 처리하는거라고 생각하면 좋을듯
    mangadataset = MangaDataset()
    mangadataset.load_Manga(dataset_dir)  # 불러옴
    mangadataset.prepare()


    COCO_WEIGHTS_PATH = "asdf.h5"  #model's path # 모델의 경로
    config = MangaInferenceConfig(mangadataset)
    model_dir = '../logs'  # 학습할때 모델 저장되는 장소
    model = modellib.MaskRCNN(mode="inference", config=config,
                          model_dir=model_dir)
    weights_path = COCO_WEIGHTS_PATH
    model.load_weights(weights_path, by_name=True)
    image, mask_list, bbox_list = detect(model, mangadataset)
    os.remove(dataset_dir + filename)
    return image,mask_list, bbox_list


""" Example """
path='../'
image,mask_list,bbox_list=extract_sfx(path)
#print(image.shape,mask_list[0].shape)


# for i in range(len(id_list)):
formask=np.zeros(image.shape)
if len(mask_list)!=0 :
    for i in range(len(mask_list)): #여러장이 가능하다면 좋겠지만 일단한장.

        # print(image.shape)
        formask[mask_list[i][:,:,0]==255] = np.asarray([255,255,255]) 

formask=formask.astype(np.float32) #이게 이미지에 아예 마스크를 덧씌운 array.
print(image.shape)
# """저장용 코드"""
r,g,b = cv.split(formask)
img_bgr = cv.merge([b,g,r])
savepath='../' #수정가능
cv.imwrite(savepath+'{}_masked.jpg'.format('white'),img_bgr)