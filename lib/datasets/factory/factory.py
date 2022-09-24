# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
from datasets.pascal_voc import pascal_voc
from datasets.cityscape import cityscape
from datasets.rtts import rtts
from datasets.foggydriving import foggydriving
from datasets.coco import coco
from datasets.imagenet import imagenet
from datasets.vg import vg
from datasets.nus import nus
from datasets.stf import stf

import numpy as np

# Set up voc_<year>_<split>
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'voc_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

for year in ['2007', '2012']:
  for split in ['test_foggy_beta0.01', 'test_foggy_beta0.005', 'train_foggy_beta0.01', 'train_foggy_beta0.005', 'train_s', 'train_t', 'train_all', 'test_s', 'test_t','test_all', 'train_dense', 'test_dense', 'train_s_quick', 'train_t_quick', 'train_dense_quick', 'train_s_partA', 'train_dense_partA', 'train_s_partB', 'train_dense_partB', 'train_night', 'test_night', 'test_night_quick', 'quicktest_dense']:
    name = 'cityscape_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: cityscape(split, year))

for year in ['2007']:
  for split in ['valid', 'valid_aug', 'valid_quick']:
    name = 'nus_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: nus(split, year))

for year in ['2007']:
  for split in ['test']:
    name = 'foggydriving_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: foggydriving(split, year))

for year in ['2007']:
  for split in ['test', 'train']:
    name = 'rtts_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: rtts(split, year))

for year in ['2007']:
  for split in ['fog_day', 'dense_fog_day', 'light_fog_day']:
    name = 'stf_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: stf(split, year))

# Set up coco_2014_<split>
for year in ['2014']:
  for split in ['train', 'val', 'minival', 'valminusminival', 'trainval']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2014_cap_<split>
for year in ['2014']:
  for split in ['train', 'val', 'capval', 'valminuscapval', 'trainval']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2015_<split>
for year in ['2015']:
  for split in ['test', 'test-dev']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up vg_<split>
# for version in ['1600-400-20']:
#     for split in ['minitrain', 'train', 'minival', 'val', 'test']:
#         name = 'vg_{}_{}'.format(version,split)
#         __sets[name] = (lambda split=split, version=version: vg(version, split))
for version in ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']:
    for split in ['minitrain', 'smalltrain', 'train', 'minival', 'smallval', 'val', 'test']:
        name = 'vg_{}_{}'.format(version,split)
        __sets[name] = (lambda split=split, version=version: vg(version, split))

# set up image net.
for split in ['train', 'val', 'val1', 'val2', 'test']:
    name = 'imagenet_{}'.format(split)
    devkit_path = 'data/imagenet/ILSVRC/devkit'
    data_path = 'data/imagenet/ILSVRC'
    __sets[name] = (lambda split=split, devkit_path=devkit_path, data_path=data_path: imagenet(split,devkit_path,data_path))

def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())
