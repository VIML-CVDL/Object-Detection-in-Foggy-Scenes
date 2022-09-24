# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time

from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

#from roi_data_layer.roidb import combined_roidb
#from roi_data_layer.roibatchLoader import roibatchLoader

from roi_da_data_layer.roidb import combined_roidb
from roi_da_data_layer.roibatchLoader import roibatchLoader


from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient

from model.da_faster_rcnn.vgg16 import vgg16
from model.da_faster_rcnn.resnet_transmission_depth_src_cst_unet_detach_p import resnet_transmission_depth_src_cst_unet_detach_p as resnet_p
import torch.nn.functional as F
from collections import OrderedDict
from model.ema.optim_weight_ema import WeightEMA

#from model.da_faster_rcnn.vgg16 import vgg16
#from model.da_faster_rcnn.resnet import resnet

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='cityscape', type=str)
  parser.add_argument('--net', dest='net',
                    help='vgg16, res101',
                    default='vgg16', type=str)
  parser.add_argument('--start_epoch', dest='start_epoch',
                      help='starting epoch',
                      default=1, type=int)
  parser.add_argument('--epochs', dest='max_epochs',
                      help='number of epochs to train',
                      default=20, type=int)
  parser.add_argument('--disp_interval', dest='disp_interval',
                      help='number of iterations to display',
                      default=100, type=int)
  parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                      help='number of iterations to display',
                      default=10000, type=int)

  parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save models', default="./models",
                      type=str)
  parser.add_argument('--nw', dest='num_workers',
                      help='number of worker to load data',
                      default=0, type=int)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')

# config optimization
  parser.add_argument('--o', dest='optimizer',
                      help='training optimizer',
                      default="sgd", type=str)
  parser.add_argument('--lr', dest='lr',
                      help='starting learning rate',
                      default=0.002, type=float)
  parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                      help='step to do learning rate decay, unit is epoch',
                      default=6, type=int)
  parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                      help='learning rate decay ratio',
                      default=0.1, type=float)
  parser.add_argument('--lamda', dest='lamda',
                      help='DA loss param',
                      default=0.1, type=float)


# set training session
  parser.add_argument('--s', dest='session',
                      help='training session',
                      default=1, type=int)

# resume trained model
  parser.add_argument('--r', dest='resume',
                      help='resume checkpoint or not',
                      default=False, type=bool)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load model',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load model',
                      default=0, type=int)
# log and diaplay
  parser.add_argument('--use_tfb', dest='use_tfboard',
                      help='whether use tensorboard',
                      action='store_true')

  parser.add_argument('--pretrained', dest='pretrained',
                      help='whether use pretrained model',
                      type=str)

  parser.add_argument('--teacher', dest='teacher',
                      help='whether use teacher model',
                      type=str)

  args = parser.parse_args()
  return args


class sampler(Sampler):
  def __init__(self, train_size, batch_size):
    self.num_data = train_size
    self.num_per_batch = int(train_size / batch_size)
    self.batch_size = batch_size
    self.range = torch.arange(0,batch_size).view(1, batch_size).long()
    self.leftover_flag = False
    if train_size % batch_size:
      self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
      self.leftover_flag = True

  def __iter__(self):
    rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
    self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

    self.rand_num_view = self.rand_num.view(-1)

    if self.leftover_flag:
      self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

    return iter(self.rand_num_view)

  def __len__(self):
    return self.num_data

if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)

  if args.dataset == "pascal_voc":
      print('loading our dataset...........')
      args.imdb_name = "voc_2007_train"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[4,8,16,32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  elif args.dataset == "cityscape":
      print('loading our dataset...........')
      args.s_imdb_name = "cityscape_2007_train_s"
      args.t_imdb_name = "cityscape_2007_train_dense"
      args.s_imdbtest_name="cityscape_2007_test_s"
      args.t_imdbtest_name="cityscape_2007_test_dense"
      args.set_cfgs = ['ANCHOR_SCALES', '[4,8,16,32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  elif args.dataset == "pascal_voc_0712":
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "coco":
      args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
      args.imdbval_name = "coco_2014_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  elif args.dataset == "imagenet":
      args.imdb_name = "imagenet_train"
      args.imdbval_name = "imagenet_val"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
  elif args.dataset == "vg":
      # train sizes: train, smalltrain, minitrain
      # train scale: ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']
      args.imdb_name = "vg_150-50-50_minitrain"
      args.imdbval_name = "vg_150-50-50_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

  args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)
  np.random.seed(cfg.RNG_SEED)

  #torch.backends.cudnn.benchmark = True
  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  # train set
  # -- Note: Use validation set and disable the flipped to enable faster loading.
  cfg.TRAIN.USE_FLIPPED = True
  cfg.USE_GPU_NMS = args.cuda

  s_imdb, s_roidb, s_ratio_list, s_ratio_index = combined_roidb(args.s_imdb_name)
  s_train_size = len(s_roidb)  # add flipped         image_index*2

  t_imdb, t_roidb, t_ratio_list, t_ratio_index = combined_roidb(args.t_imdb_name)
  t_train_size = len(t_roidb)  # add flipped         image_index*2

  print('source {:d} target {:d} roidb entries'.format(len(s_roidb),len(t_roidb)))

  output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  s_sampler_batch = sampler(s_train_size, args.batch_size)
  t_sampler_batch=sampler(t_train_size,args.batch_size)

  s_dataset = roibatchLoader(s_roidb, s_ratio_list, s_ratio_index, args.batch_size, \
                           s_imdb.num_classes, training=True, transmission=True, depth=True, aug=False, clean=False)

  s_dataloader = torch.utils.data.DataLoader(s_dataset, batch_size=args.batch_size,
                            sampler=s_sampler_batch, num_workers=args.num_workers)

  t_dataset =roibatchLoader(t_roidb, t_ratio_list, t_ratio_index, args.batch_size, \
                           t_imdb.num_classes, training=False, transmission=True, depth=True, aug=False, clean=True)

  t_dataloader = torch.utils.data.DataLoader(t_dataset, batch_size=args.batch_size,
                                           sampler=t_sampler_batch, num_workers=args.num_workers)
  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_data_clean = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)
  transmission_map = torch.FloatTensor(1)
  depth = torch.FloatTensor(1)
  clean = torch.FloatTensor(1)
  need_backprop = torch.FloatTensor(1)

  tgt_im_data = torch.FloatTensor(1)
  tgt_im_data_clean = torch.FloatTensor(1)
  tgt_im_info = torch.FloatTensor(1)
  tgt_num_boxes = torch.LongTensor(1)
  tgt_gt_boxes = torch.FloatTensor(1)
  tgt_transmission_map = torch.FloatTensor(1)
  tgt_depth = torch.FloatTensor(1)
  tgt_clean = torch.FloatTensor(1)
  tgt_need_backprop = torch.FloatTensor(1)


  # ship to cuda
  if args.cuda:
      im_data = im_data.cuda()
      im_data_clean = im_data_clean.cuda()
      im_info = im_info.cuda()
      num_boxes = num_boxes.cuda()
      gt_boxes = gt_boxes.cuda()
      transmission_map = transmission_map.cuda()
      depth = depth.cuda()
      clean = clean.cuda()
      need_backprop = need_backprop.cuda()

      tgt_im_data = tgt_im_data.cuda()
      tgt_im_data_clean = tgt_im_data_clean.cuda()
      tgt_im_info = tgt_im_info.cuda()
      tgt_num_boxes = tgt_num_boxes.cuda()
      tgt_gt_boxes = tgt_gt_boxes.cuda()
      tgt_transmission_map = tgt_transmission_map.cuda()
      tgt_depth = tgt_depth.cuda()
      tgt_clean = tgt_clean.cuda()
      tgt_need_backprop = tgt_need_backprop.cuda()

  # make variable
  im_data = Variable(im_data)
  im_data_clean = Variable(im_data_clean)
  im_info = Variable(im_info)
  num_boxes = Variable(num_boxes)
  gt_boxes = Variable(gt_boxes)
  transmission_map = Variable(transmission_map)
  depth = Variable(depth)
  clean = Variable(clean)
  need_backprop = Variable(need_backprop)

  tgt_im_data = Variable(tgt_im_data)
  tgt_im_data_clean = Variable(tgt_im_data_clean)
  tgt_im_info = Variable(tgt_im_info)
  tgt_num_boxes = Variable(tgt_num_boxes)
  tgt_gt_boxes = Variable(tgt_gt_boxes)
  tgt_transmission_map = Variable(tgt_transmission_map)
  tgt_depth = Variable(tgt_depth)
  tgt_clean = Variable(tgt_clean)
  tgt_need_backprop = Variable(tgt_need_backprop)

  if args.cuda:
    cfg.CUDA = True

  # initilize the network here.
  if args.net == 'vgg16':
    fasterRCNN = vgg16(s_imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res101':
    fasterRCNN = resnet_p(s_imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
    fasterRCNN_teacher = resnet_p(s_imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res50':
    fasterRCNN = resnetDA(s_imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res152':
    fasterRCNN = resnetDA(s_imdb.classes, 152, pretrained=True, class_agnostic=args.class_agnostic)
  else:
    print("network is not defined")
    pdb.set_trace()

  fasterRCNN.create_architecture()
  fasterRCNN_teacher.create_architecture()
  if args.pretrained:
      checkpoint = torch.load(args.pretrained)
      fasterRCNN.load_state_dict(checkpoint['model'])

  lr = cfg.TRAIN.LEARNING_RATE
  lr = args.lr
  #tr_momentum = cfg.TRAIN.MOMENTUM
  #tr_momentum = args.momentum

  params = []
  student_detection_params = []
  for key, value in dict(fasterRCNN.named_parameters()).items():
    if value.requires_grad:
      if 'bias' in key:
        params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
      else:
        params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
      student_detection_params += [value]

  teacher_detection_params = []
  for key, value in dict(fasterRCNN_teacher.named_parameters()).items():
    if value.requires_grad:
        teacher_detection_params += [value]
        value.requires_grad = False

  if args.optimizer == "adam":
    #lr = lr * 0.1
    optimizer = torch.optim.Adam(params)

  elif args.optimizer == "sgd":
    optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

  teacher_optimizer = WeightEMA(
    teacher_detection_params, student_detection_params, alpha=0.99
  )

  if args.resume:
    load_name = os.path.join(output_dir,
      'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
    print("loading checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    args.session = checkpoint['session']
    args.start_epoch = checkpoint['epoch']
    fasterRCNN.load_state_dict(checkpoint['model'])
    #optimizer.load_state_dict(checkpoint['optimizer'])
    #lr = optimizer.param_groups[0]['lr']
    if 'pooling_mode' in checkpoint.keys():
      cfg.POOLING_MODE = checkpoint['pooling_mode']
    print("loaded checkpoint %s" % (load_name))


    load_name = os.path.join(output_dir,
      'faster_rcnn_{}_{}_{}_teacher.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
    print("loading checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    fasterRCNN_teacher.load_state_dict(checkpoint['model'])


  if args.mGPUs:
    fasterRCNN = nn.DataParallel(fasterRCNN)

  if args.cuda:
    fasterRCNN.cuda()
    fasterRCNN_teacher.cuda()

  iters_per_epoch = int(s_train_size / args.batch_size)

  if args.use_tfboard:
    from tensorboardX import SummaryWriter
    logger = SummaryWriter("logs")

  pretrained_epoch = 2

  for epoch in range(args.start_epoch, args.max_epochs + 1):
    # setting to train mode

    fasterRCNN.train()
    fasterRCNN_teacher.train()
    loss_temp = 0
    start = time.time()

    if epoch % (args.lr_decay_step + 1) == 0:
        adjust_learning_rate(optimizer, args.lr_decay_gamma)
        lr *= args.lr_decay_gamma

    data_iter = iter(s_dataloader)
    tgt_data_iter=iter(t_dataloader)

    for step in range(iters_per_epoch):
      data = next(data_iter)
      tgt_data=next(tgt_data_iter)

      im_data.data.resize_(data[0].size()).copy_(data[0])  # change holder size
      im_info.data.resize_(data[1].size()).copy_(data[1])
      gt_boxes.data.resize_(data[2].size()).copy_(data[2])
      num_boxes.data.resize_(data[3].size()).copy_(data[3])
      need_backprop.data.resize_(data[4].size()).copy_(data[4])
      transmission_map.data.resize_(data[6].size()).copy_(data[6])
      depth.data.resize_(data[7].size()).copy_(data[7])
      need_backprop[:]=1
      tgt_im_data.data.resize_(tgt_data[0].size()).copy_(tgt_data[0])  # change holder size
      tgt_im_info.data.resize_(tgt_data[1].size()).copy_(tgt_data[1])
      tgt_gt_boxes.data.resize_(tgt_data[2].size()).copy_(tgt_data[2])
      tgt_num_boxes.data.resize_(tgt_data[3].size()).copy_(tgt_data[3])
      tgt_need_backprop.data.resize_(tgt_data[4].size()).copy_(tgt_data[4])
      tgt_transmission_map.data.resize_(tgt_data[6].size()).copy_(tgt_data[6])
      tgt_depth.data.resize_(tgt_data[7].size()).copy_(tgt_data[7])
      tgt_clean.data.resize_(tgt_data[8].size()).copy_(tgt_data[8])


      """student"""
      fasterRCNN.zero_grad()
      rois, cls_prob, bbox_pred, \
      rpn_loss_cls, rpn_loss_box, \
      RCNN_loss_cls, RCNN_loss_bbox, \
      rois_label,prior_loss,dep_loss,cst_loss,rec_loss=\
          fasterRCNN(im_data, im_info, gt_boxes, num_boxes,need_backprop,
                     tgt_im_data, tgt_im_info, tgt_gt_boxes, tgt_num_boxes, tgt_need_backprop, transmission_map, tgt_transmission_map, depth, tgt_depth, tgt_clean)


      loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
           + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()\
           + prior_loss.mean() + 10*dep_loss.mean() + cst_loss.mean() + rec_loss.mean()

      if epoch > pretrained_epoch:
          fasterRCNN_teacher.eval()
          rois_teacher, cls_prob_teacher, bbox_pred_teacher, \
          _, _,  _, _, rois_label_teacher = fasterRCNN_teacher(tgt_clean, tgt_im_info, tgt_gt_boxes, tgt_num_boxes, teacher=True)

          scores_teacher = cls_prob_teacher.data
          boxes_teacher = rois_teacher.data[:, :, 1:5]

          if cfg.TEST.BBOX_REG:
              # Apply bounding-box regression deltas
              box_deltas_teacher = bbox_pred_teacher.data
              if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
              # Optionally normalize targets by a precomputed mean and stdev
                if args.class_agnostic:
                    box_deltas_teacher = box_deltas_teacher.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas_teacher = box_deltas_teacher.view(1, -1, 4)
                else:
                    box_deltas_teacher = box_deltas_teacher.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas_teacher = box_deltas_teacher.view(1, -1, 4 * len(t_imdb.classes))

              pred_boxes_teacher = bbox_transform_inv(boxes_teacher, box_deltas_teacher, 1)
              pred_boxes_teacher = clip_boxes(pred_boxes_teacher, tgt_im_info.data, 1)
          else:
            # Simply repeat the boxes, once for each class
            pred_boxes_teacher = np.tile(boxes_teacher, (1, scores_teacher.shape[1]))

          scores_teacher = scores_teacher.squeeze()
          pred_boxes_teacher = pred_boxes_teacher.squeeze()
          gt_boxes_teacher_target = []
          pre_thresh = 0.0
          thresh = 0.8
          empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))
          for j in range(1, len(t_imdb.classes)):
            inds = torch.nonzero(scores_teacher[:, j] > pre_thresh).view(-1)
            # if there is det
            if inds.numel() > 0:
                cls_scores_teacher = scores_teacher[:, j][inds]
                _, order = torch.sort(cls_scores_teacher, 0, True)
                if args.class_agnostic:
                    cls_boxes_teacher = pred_boxes_teacher[inds, :]
                else:
                    cls_boxes_teacher = pred_boxes_teacher[inds][:, j * 4 : (j + 1) * 4]

                cls_dets = torch.cat((cls_boxes_teacher, cls_scores_teacher.unsqueeze(1)), 1)
                # cls_dets = torch.cat((cls_boxes_teacher, cls_scores_teacher), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_dets, cfg.TEST.NMS)
                cls_dets = cls_dets[keep.view(-1).long()]
                # all_boxes_teacher[j][i] = cls_dets.cpu().numpy()
                cls_dets_numpy = cls_dets.cpu().numpy()
                for i in range(np.minimum(10, cls_dets_numpy.shape[0])):
                    bbox = tuple(
                        int(np.round(x)) for x in cls_dets_numpy[i, :4]
                    )
                    score = cls_dets_numpy[i, -1]
                    if score > thresh:
                        gt_boxes_teacher_target.append(list(bbox[0:4]) + [j])

          gt_boxes_teacher_padding = torch.FloatTensor(cfg.MAX_NUM_GT_BOXES, 5).zero_()
          if len(gt_boxes_teacher_target) != 0:
            gt_boxes_teacher_numpy = torch.FloatTensor(gt_boxes_teacher_target)
            num_boxes_teacher_cpu = torch.LongTensor(
                [min(gt_boxes_teacher_numpy.size(0), cfg.MAX_NUM_GT_BOXES)]
            )
            gt_boxes_teacher_padding[:num_boxes_teacher_cpu, :] = gt_boxes_teacher_numpy[:num_boxes_teacher_cpu]
          else:
            num_boxes_teacher_cpu = torch.LongTensor([0])

          gt_boxes_teacher_padding = torch.unsqueeze(gt_boxes_teacher_padding, 0)
          tgt_gt_boxes.data.resize_(gt_boxes_teacher_padding.size()).copy_(gt_boxes_teacher_padding)
          tgt_num_boxes.data.resize_(num_boxes_teacher_cpu.size()).copy_(num_boxes_teacher_cpu)


          """   faster-rcnn loss + DA loss for source and DA loss for target """
          _, _, _, \
          rpn_loss_cls_p, rpn_loss_box_p, \
          RCNN_loss_cls_p, RCNN_loss_bbox_p=\
              fasterRCNN(tgt_im_data, tgt_im_info, tgt_gt_boxes, tgt_num_boxes, da=False)

          loss += 0.01 * (
              rpn_loss_cls_p.mean()
              + rpn_loss_box_p.mean()
              + RCNN_loss_cls_p.mean()
              + RCNN_loss_bbox_p.mean()
          )

      loss_temp += loss.item()

      # backward
      optimizer.zero_grad()
      loss.backward()
      if args.net == "vgg16":
          clip_gradient(fasterRCNN, 10.)
      optimizer.step()
      fasterRCNN_teacher.zero_grad()
      teacher_optimizer.step()

      if step % args.disp_interval == 0:
        end = time.time()
        if step > 0:
          loss_temp /= (args.disp_interval + 1)

        if args.mGPUs:
          loss_rpn_cls = rpn_loss_cls.mean().item()
          loss_rpn_box = rpn_loss_box.mean().item()
          loss_rcnn_cls = RCNN_loss_cls.mean().item()
          loss_rcnn_box = RCNN_loss_bbox.mean().item()
          fg_cnt = torch.sum(rois_label.data.ne(0))
          bg_cnt = rois_label.data.numel() - fg_cnt
        else:
          loss_rpn_cls = rpn_loss_cls.item()
          loss_rpn_box = rpn_loss_box.item()
          loss_rcnn_cls = RCNN_loss_cls.item()
          loss_rcnn_box = RCNN_loss_bbox.item()
          loss_prior=prior_loss.item()
          loss_dep = 10*dep_loss.item()
          loss_rec = rec_loss.item()
          loss_cst = cst_loss.item()
          if epoch > pretrained_epoch:
              loss_tgt_rpn_loss = rpn_loss_cls_p.item() + rpn_loss_box_p.item()
              loss_tgt_RCNN_loss = RCNN_loss_cls_p.item() + RCNN_loss_bbox_p.item()
          else:
              loss_tgt_rpn_loss = 0
              loss_tgt_RCNN_loss = 0
          fg_cnt = torch.sum(rois_label.data.ne(0))
          bg_cnt = rois_label.data.numel() - fg_cnt

        print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                                % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
        print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end-start))
        print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f, loss_prior %.4f, loss_dep %.4f, loss_rec %.4f, loss_tgt_rpn_loss %.4f, loss_tgt_RCNN_loss %.4f" \
                      % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box,loss_prior,loss_dep,loss_rec, loss_tgt_rpn_loss, loss_tgt_RCNN_loss))
        if args.use_tfboard:
          info = {
            'loss': loss_temp,
            'loss_rpn_cls': loss_rpn_cls,
            'loss_rpn_box': loss_rpn_box,
            'loss_rcnn_cls': loss_rcnn_cls,
            'loss_rcnn_box': loss_rcnn_box,
            'loss_prior': loss_prior,
            'loss_dep': loss_dep,
            'loss_rec': loss_rec,
            'loss_cst': loss_cst,
            'loss_tgt_rpn_loss': loss_tgt_rpn_loss,
            'loss_tgt_RCNN_loss': loss_tgt_RCNN_loss,
          }
          logger.add_scalars("logs_s_{}/losses".format(args.session), info, (epoch - 1) * iters_per_epoch + step)

        loss_temp = 0
        start = time.time()

    save_name = os.path.join(output_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.session, epoch, step))
    save_checkpoint({
      'session': args.session,
      'epoch': epoch + 1,
      'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
      'optimizer': optimizer.state_dict(),
      'pooling_mode': cfg.POOLING_MODE,
      'class_agnostic': args.class_agnostic,
    }, save_name)
    print('save model: {}'.format(save_name))


    save_name = os.path.join(output_dir, 'faster_rcnn_{}_{}_{}_teacher.pth'.format(args.session, epoch, step))
    save_checkpoint({
      'session': args.session,
      'epoch': epoch + 1,
      'model': fasterRCNN_teacher.module.state_dict() if args.mGPUs else fasterRCNN_teacher.state_dict(),
      'optimizer': optimizer.state_dict(),
      'pooling_mode': cfg.POOLING_MODE,
      'class_agnostic': args.class_agnostic,
    }, save_name)
    print('save model: {}'.format(save_name))
  if args.use_tfboard:
    logger.close()
