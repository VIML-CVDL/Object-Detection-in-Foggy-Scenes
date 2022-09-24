# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model.utils.config import cfg
import math
import torchvision.models as models
from model.da_faster_rcnn.faster_rcnn import _fasterRCNN
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta
from model.da_faster_rcnn.DA import _ImageDA
from model.da_faster_rcnn.DA import _InstanceDA

def conv3x3(in_planes, out_planes, stride=1, padding=0):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
           padding=padding, bias=False)
def conv1x1(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
           padding=0, bias=False)

class MergeRes(nn.Module):
    def __init__(self, in_channels, out_channels, context=False):
        super(MergeRes, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, padding=1)
        self.context = context
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        if self.context:
          feat = x
        if self.context:
          return x,feat#torch.cat((feat1,feat2),1)#F
        else:
          return x

class vgg16_ms3_phy_concat(_fasterRCNN):
  def __init__(self, classes, pretrained=False, class_agnostic=False):
    self.model_path = 'data/pretrained_model/vgg16_caffe.pth'
    self.dout_base_model = 512
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic
    _fasterRCNN.__init__(self, classes, class_agnostic)

  def _init_modules(self):
    vgg = models.vgg16()
    if self.pretrained:
        print("Loading pretrained weights from %s" %(self.model_path))
        state_dict = torch.load(self.model_path)
        vgg.load_state_dict({k:v for k,v in state_dict.items() if k in vgg.state_dict()})

    vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])

    self.RCNN_base1 = nn.Sequential(*list(vgg.features._modules.values())[:-22])
    self.RCNN_base2 = nn.Sequential(*list(vgg.features._modules.values())[-22:-15])
    self.RCNN_base3 = nn.Sequential(*list(vgg.features._modules.values())[-15:-8])
    self.RCNN_base4 = nn.Sequential(*list(vgg.features._modules.values())[-8:-1])

    self.RCNN_imageDA = _ImageDA(self.dout_base_model)
    self.RCNN_imageDA_1 = _ImageDA(128)
    self.RCNN_imageDA_2 = _ImageDA(256)
    self.RCNN_imageDA_3 = _ImageDA(self.dout_base_model)

    self.mergeres = MergeRes(256, 128)
    # Fix the layers before conv3:
    for layer in range(5):
      for p in self.RCNN_base1[layer].parameters(): p.requires_grad = False

    # self.RCNN_base = _RCNN_base(vgg.features, self.classes, self.dout_base_model)

    self.RCNN_top = vgg.classifier

    # not using the last maxpool layer
    self.RCNN_cls_score = nn.Linear(4096, self.n_classes)

    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(4096, 4)
    else:
      self.RCNN_bbox_pred = nn.Linear(4096, 4 * self.n_classes)

  def forward(self, im_data, im_info, gt_boxes, num_boxes, vis=False, roivis=False):

    batch_size = im_data.size(0)
    im_info = im_info.data     #(size1,size2, image ratio(new image / source image) )
    gt_boxes = gt_boxes.data
    num_boxes = num_boxes.data

    '''----------stage 1-----------------------'''
    # feed image data to base model to obtain base feature map
    base_feat_rgb = self.RCNN_base1(im_data)
    red = im_data[:, 0:1, :, :]
    red = torch.cat((red, im_data[:, 0:1, :, :]), 1)
    red = torch.cat((red, im_data[:, 0:1, :, :]), 1)
    green = im_data[:, 1:2, :, :]
    green = torch.cat((green, im_data[:, 1:2, :, :]), 1)
    green = torch.cat((green, im_data[:, 1:2, :, :]), 1)
    blue = im_data[:, 2:3, :, :]
    blue = torch.cat((blue, im_data[:, 2:3, :, :]), 1)
    blue = torch.cat((blue, im_data[:, 2:3, :, :]), 1)
    base_feat_red = self.RCNN_base1(red)
    base_feat_green = self.RCNN_base1(green)
    base_feat_blue = self.RCNN_base1(blue)
    b, c, h, w = base_feat_red.size()
    Rflat = torch.reshape(base_feat_red, (b, 1, c, h*w))
    Gflat = torch.reshape(base_feat_green, (b, 1, c, h*w))
    Bflat = torch.reshape(base_feat_blue, (b, 1, c, h*w))
    stack_tensor = torch.cat((Rflat, Gflat, Bflat), dim=1)
    max_tensor, _ = torch.max(stack_tensor, 1)
    min_tensor, _ = torch.min(stack_tensor, 1)
    mul_max_feat = torch.reshape(max_tensor, (b, c, h, w))
    mul_min_feat = torch.reshape(min_tensor, (b, c, h, w))
    max_flat = torch.reshape(mul_max_feat, (b, 1, c, h*w))
    min_flat = torch.reshape(mul_min_feat, (b, 1, c, h*w))
    res_tensor = max_flat - min_flat
    res_feat = torch.reshape(res_tensor, (b, c, h, w))
    base_feat_1 = torch.cat((base_feat_rgb, res_feat), 1)
    base_feat_1 = self.mergeres(base_feat_1)

    '''----------stage 2------------------------'''
    base_feat_2 = self.RCNN_base2(base_feat_1)

    '''----------stage 3------------------------'''
    base_feat_3 = self.RCNN_base3(base_feat_2)

    '''----------stage 4------------------------'''
    base_feat = self.RCNN_base4(base_feat_3)

    if vis:
        return base_feat

    # feed base feature map tp RPN to obtain rois
    rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

    # if it is training phrase, then use ground trubut bboxes for refining
    if self.training:
        roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
        rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

        rois_label = Variable(rois_label.view(-1).long())
        rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
        rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
        rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
    else:
        rois_label = None
        rois_target = None
        rois_inside_ws = None
        rois_outside_ws = None
        rpn_loss_cls = 0
        rpn_loss_bbox = 0

    rois = Variable(rois)
    # do roi pooling based on predicted rois

    if cfg.POOLING_MODE == 'crop':
        # pdb.set_trace()
        # pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
        grid_xy = _affine_grid_gen(rois.view(-1, 5), base_feat.size()[2:], self.grid_size)
        grid_yx = torch.stack([grid_xy.data[:,:,:,1], grid_xy.data[:,:,:,0]], 3).contiguous()
        pooled_feat = self.RCNN_roi_crop(base_feat, Variable(grid_yx).detach())
        if cfg.CROP_RESIZE_WITH_MAX_POOL:
            pooled_feat = F.max_pool2d(pooled_feat, 2, 2)
    elif cfg.POOLING_MODE == 'align':
        if roivis:
            gt_rois = gt_boxes[0]
            gt_rois = gt_rois[gt_rois[:,0]!=0]
            gt_rois = torch.cat((torch.index_select(gt_rois, 1, torch.cuda.LongTensor([4])), torch.index_select(gt_rois, 1, torch.cuda.LongTensor([0,1,2,3]))), dim=1)
            gt_labels = gt_rois[:,0].clone()
            for k in range(len(gt_rois[:,0])):
                gt_rois[k,0] = 0
            pooled_feat = self.RCNN_roi_align(base_feat, gt_rois)
            return pooled_feat, gt_labels
        pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
    elif cfg.POOLING_MODE == 'pool':
        pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))

    # feed pooled features to top model
    pooled_feat = self._head_to_tail(pooled_feat)

    # compute bbox offset
    bbox_pred = self.RCNN_bbox_pred(pooled_feat)
    if self.training and not self.class_agnostic:
        # select the corresponding columns according to roi labels
        bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
        bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
        bbox_pred = bbox_pred_select.squeeze(1)

    # compute object classification probability
    cls_score = self.RCNN_cls_score(pooled_feat)
    cls_prob = F.softmax(cls_score, 1)

    RCNN_loss_cls = 0
    RCNN_loss_bbox = 0

    if self.training:
        # classification loss
        RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

        # bounding box regression L1 loss
        RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)


    cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
    bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

    return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label

  def _head_to_tail(self, pool5):

    pool5_flat = pool5.view(pool5.size(0), -1)
    fc7 = self.RCNN_top(pool5_flat)

    return fc7
