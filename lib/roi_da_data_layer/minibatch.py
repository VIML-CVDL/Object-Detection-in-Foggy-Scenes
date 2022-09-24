# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
from imageio import imread
from model.utils.config import cfg
from model.utils.blob import prep_im_for_blob, im_list_to_blob
import cv2
import pdb
from model.utils.aug_utils import build_strong_augmentation
import numpy as np
from PIL import Image


def get_minibatch(roidb, num_classes, transmission=False, transmission_ms=False, transmission_ms3=False, depth=False, grad=False, clean=False, aug=False):
  """Given a roidb, construct a minibatch sampled from it."""
  num_images = len(roidb)
  # Sample random scales to use for each image in this batch
  random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                  size=num_images)
  assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
    'num_images ({}) must divide BATCH_SIZE ({})'. \
    format(num_images, cfg.TRAIN.BATCH_SIZE)

  # Get the input image blob, formatted for caffe
  im_blob, im_scales = _get_image_blob(roidb, random_scale_inds, False)

  blobs = {'data': im_blob}

  im_name=roidb[0]['image']
  if im_name.find('_s') == -1:  # target domain
    blobs['need_backprop']=np.zeros((1,),dtype=np.float32)
  else:
    blobs['need_backprop']=np.ones((1,),dtype=np.float32)


  assert len(im_scales) == 1, "Single batch only"
  assert len(roidb) == 1, "Single batch only"

  # gt boxes: (x1, y1, x2, y2, cls)
  if cfg.TRAIN.USE_ALL_GT:
    # Include all ground truth boxes
    gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
  else:
    # For the COCO ground truth boxes, exclude the ones that are ''iscrowd''
    gt_inds = np.where((roidb[0]['gt_classes'] != 0) & np.all(roidb[0]['gt_overlaps'].toarray() > -1.0, axis=1))[0]
  gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
  gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
  gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
  blobs['gt_boxes'] = gt_boxes
  blobs['im_info'] = np.array(
    [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
    dtype=np.float32)

  blobs['im_name'] = im_name

  if transmission:
    transmission_blob, transmission_scales = _get_transmission_blob(roidb)
    blobs['transmission'] = transmission_blob
  if transmission_ms:
    transmission_blob_ms, transmission_scales_ms = _get_transmission_blob(roidb, ms=True)
    blobs['transmission_ms'] = transmission_blob_ms
  if transmission_ms3:
    transmission_blob_ms3, transmission_scales_ms3 = _get_transmission_blob(roidb, ms3=True)
    blobs['transmission_ms3'] = transmission_blob_ms3
  if depth:
    depth_blob, depth_scales = _get_depth_blob(roidb)
    blobs['depth'] = depth_blob
    depth_blob2, depth_scales2 = _get_depth_blob2(roidb)
    blobs['depth2'] = depth_blob2
  if grad:
    grad_blob, grad_scales = _get_grad_blob(roidb)
    blobs['grad'] = grad_blob
  if clean:
    clean_blob, clean_scales = _get_clean_blob(roidb, random_scale_inds)
    blobs['clean'] = clean_blob

  if aug:

        # Get the input image blob, formatted for caffe
        im_blob, im_scales = _get_image_blob(roidb, random_scale_inds, aug)
        blobs_aug = {'data': im_blob}

        im_name=roidb[0]['image']
        if im_name.find('_s') == -1:  # target domain
          blobs_aug['need_backprop']=np.zeros((1,),dtype=np.float32)
        else:
          blobs_aug['need_backprop']=np.ones((1,),dtype=np.float32)


        assert len(im_scales) == 1, "Single batch only"
        assert len(roidb) == 1, "Single batch only"

        # gt boxes: (x1, y1, x2, y2, cls)
        if cfg.TRAIN.USE_ALL_GT:
          # Include all ground truth boxes
          gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
        else:
          # For the COCO ground truth boxes, exclude the ones that are ''iscrowd''
          gt_inds = np.where((roidb[0]['gt_classes'] != 0) & np.all(roidb[0]['gt_overlaps'].toarray() > -1.0, axis=1))[0]
        gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
        gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
        gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
        blobs_aug['gt_boxes'] = gt_boxes
        blobs_aug['im_info'] = np.array(
          [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
          dtype=np.float32)

        blobs_aug['im_name'] = im_name

        return blobs, blobs_aug
  return blobs

def _get_image_blob(roidb, scale_inds, aug):
  """Builds an input blob from the images in the roidb at the specified
  scales.
  """
  num_images = len(roidb)
  processed_ims = []
  im_scales = []
  for i in range(num_images):
    #im = cv2.imread(roidb[i]['image'])
    im = imread(roidb[i]['image'])
    if aug:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(im)
        strong_aug = build_strong_augmentation(True)
        im = np.array(strong_aug(im))
    if len(im.shape) == 2:
      im = im[:,:,np.newaxis]
      im = np.concatenate((im,im,im), axis=2)
    # flip the channel, since the original one using cv2
    # rgb -> bgr
    im = im[:,:,::-1]
    #im = cv2.resize(im, (1024, 2048),
    #                interpolation=cv2.INTER_AREA)

    if roidb[i]['flipped']:
      im = im[:, ::-1, :]
    target_size = cfg.TRAIN.SCALES[scale_inds[i]]
    im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                    cfg.TRAIN.MAX_SIZE)
    im_scales.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, im_scales


def _get_clean_blob(roidb, scale_inds):
  """Builds an input blob from the images in the roidb at the specified
  scales.
  """
  num_images = len(roidb)
  processed_ims = []
  im_scales = []
  for i in range(num_images):
    #im = cv2.imread(roidb[i]['image'])
    try:
        im = imread(roidb[i]['clean'])
    except:
        im = imread(roidb[i]['image'])

    if len(im.shape) == 2:
      im = im[:,:,np.newaxis]
      im = np.concatenate((im,im,im), axis=2)
    # flip the channel, since the original one using cv2
    # rgb -> bgr
    im = im[:,:,::-1]
    #im = cv2.resize(im, (1024, 2048),
    #                interpolation=cv2.INTER_AREA)

    if roidb[i]['flipped']:
      im = im[:, ::-1, :]
    target_size = cfg.TRAIN.SCALES[scale_inds[i]]
    im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                    cfg.TRAIN.MAX_SIZE)
    im_scales.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, im_scales


def _get_transmission_blob(roidb, ms=False, ms3=False):
  """Builds an input blob from the images in the roidb at the specified
  scales.
  """
  num_images = len(roidb)

  processed_ims = []
  im_scales = []
  for i in range(num_images):
    #im = cv2.imread(roidb[i]['image'])
    try:
        im = imread(roidb[i]['transmission'])
    except:
        im = imread(roidb[i]['image'])


    if len(im.shape) == 2:
      im = im[:,:,np.newaxis]
      im = np.concatenate((im,im,im), axis=2)
    # flip the channel, since the original one using cv2
    # rgb -> bgr
    im = im[:,:,::-1]

    if roidb[i]['flipped']:
      im = im[:, ::-1, :]
    im = im.astype('float64')
    im = cv2.normalize(im, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    im = im.astype(np.float32, copy=False)
    if ms:
        im = cv2.resize(im, (144, 69),
                        interpolation=cv2.INTER_AREA)
    elif ms3:
        im = cv2.resize(im, (294, 144),
                        interpolation=cv2.INTER_AREA)
    else:
        im = cv2.resize(im, (69, 32),
                        interpolation=cv2.INTER_AREA)

    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, im_scales

def _get_depth_blob(roidb):
  """Builds an input blob from the images in the roidb at the specified
  scales.
  """
  num_images = len(roidb)

  processed_ims = []
  im_scales = []
  for i in range(num_images):
    #im = cv2.imread(roidb[i]['image'])
    #im = cv2.imread(roidb[i]['image'])
    try:
        im = imread(roidb[i]['depth'])
    except:
        im = imread(roidb[i]['image'])


    if len(im.shape) == 2:
      im = im[:,:,np.newaxis]
      im = np.concatenate((im,im,im), axis=2)
    # flip the channel, since the original one using cv2
    # rgb -> bgr
    im = im[:,:,::-1]

    if roidb[i]['flipped']:
      im = im[:, ::-1, :]
    im = im.astype('float64')
    im = cv2.normalize(im, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    im = im.astype(np.float32, copy=False)
    im = cv2.resize(im, (69, 32),
                        interpolation=cv2.INTER_AREA)

    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, im_scales

def _get_depth_blob2(roidb):
  """Builds an input blob from the images in the roidb at the specified
  scales.
  """
  num_images = len(roidb)

  processed_ims = []
  im_scales = []
  for i in range(num_images):
    #im = cv2.imread(roidb[i]['image'])
    im = imread(roidb[i]['depth'])

    if len(im.shape) == 2:
      im = im[:,:,np.newaxis]
      im = np.concatenate((im,im,im), axis=2)
    # flip the channel, since the original one using cv2
    # rgb -> bgr
    im = im[:,:,::-1]

    if roidb[i]['flipped']:
      im = im[:, ::-1, :]
    im = im.astype('float64')
    im = cv2.normalize(im, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    im = im.astype(np.float32, copy=False)
    im = cv2.resize(im, (75, 37),
                        interpolation=cv2.INTER_AREA)

    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, im_scales

def _get_grad_blob(roidb):
  """Builds an input blob from the images in the roidb at the specified
  scales.
  """
  num_images = len(roidb)

  processed_ims = []
  im_scales = []
  for i in range(num_images):
    #im = cv2.imread(roidb[i]['image'])
    im = imread(roidb[i]['grad'])

    if len(im.shape) == 2:
      im = im[:,:,np.newaxis]
      im = np.concatenate((im,im,im), axis=2)
    # flip the channel, since the original one using cv2
    # rgb -> bgr
    im = im[:,:,::-1]

    if roidb[i]['flipped']:
      im = im[:, ::-1, :]
    im = im.astype('float64')
    im = cv2.normalize(im, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    im = im.astype(np.float32, copy=False)
    im = cv2.resize(im, (593, 297),
                        interpolation=cv2.INTER_AREA)

    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, im_scales
