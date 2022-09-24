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
from model.da_faster_rcnn.DA import _ImageDA
from model.da_faster_rcnn.DA import _InstanceDA
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, \
    _affine_theta,grad_reverse, \
    prob2entropy, self_entropy, global_attention, prob2entropy2
from model.utils.net_utils import weights_normal_init, \
    FocalLoss, sampler, calc_supp, EFocalLoss, CrossEntropy, \
    prob2entropy, \
    get_gc_discriminator


def conv3x3(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
           padding=1, bias=False)
def conv1x1(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
           padding=0, bias=False)

class netD_forward1(nn.Module):
    def __init__(self):
        super(netD_forward1, self).__init__()
        self.conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1,
                  padding=1, bias=False)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self._init_weights()
    def _init_weights(self):
      def normal_init(m, mean, stddev, truncated=False):
        """
        weight initalizer: truncated normal and random normal.
        """
        # x is a parameter
        if truncated:
          m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
        else:
          m.weight.data.normal_(mean, stddev)
          #m.bias.data.zero_()
      normal_init(self.conv1, 0, 0.01)
      normal_init(self.conv2, 0, 0.01)
      normal_init(self.conv3, 0, 0.01)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        feat = F.avg_pool2d(x, (x.size(2), x.size(3)))
        return feat

class netD_forward2(nn.Module):
    def __init__(self):
        super(netD_forward2, self).__init__()
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, stride=1,
                  padding=1, bias=False)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self._init_weights()
    def _init_weights(self):
      def normal_init(m, mean, stddev, truncated=False):
        """
        weight initalizer: truncated normal and random normal.
        """
        # x is a parameter
        if truncated:
          m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
        else:
          m.weight.data.normal_(mean, stddev)
          #m.bias.data.zero_()
      normal_init(self.conv1, 0, 0.01)
      normal_init(self.conv2, 0, 0.01)
      normal_init(self.conv3, 0, 0.01)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        feat = F.avg_pool2d(x, (x.size(2), x.size(3)))
        return feat

class netD_forward3(nn.Module):
    def __init__(self):
        super(netD_forward3, self).__init__()
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, stride=1,
                  padding=1, bias=False)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self._init_weights()
    def _init_weights(self):
      def normal_init(m, mean, stddev, truncated=False):
        """
        weight initalizer: truncated normal and random normal.
        """
        # x is a parameter
        if truncated:
          m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
        else:
          m.weight.data.normal_(mean, stddev)
          #m.bias.data.zero_()
      normal_init(self.conv1, 0, 0.01)
      normal_init(self.conv2, 0, 0.01)
      normal_init(self.conv3, 0, 0.01)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        feat = F.avg_pool2d(x, (x.size(2), x.size(3)))
        return feat


class netD_inst(nn.Module):
  def __init__(self, fc_size=2048):
    super(netD_inst, self).__init__()
    self.fc_1_inst = nn.Linear(fc_size, 1024)
    self.fc_2_inst = nn.Linear(1024, 256)
    self.fc_3_inst = nn.Linear(256, 2)
    self.relu = nn.ReLU(inplace=True)
    #self.softmax = nn.Softmax()
    #self.logsoftmax = nn.LogSoftmax()
    # self.bn = nn.BatchNorm1d(128)
    self.bn2 = nn.BatchNorm1d(2)

  def forward(self, x):
    x = self.relu(self.fc_1_inst(x))
    x = self.relu((self.fc_2_inst(x)))
    x = self.relu(self.bn2(self.fc_3_inst(x)))
    return x

class netD1(nn.Module):
    def __init__(self,context=False):
        super(netD1, self).__init__()
        self.conv1 = nn.Conv2d(256, 256, kernel_size=1, stride=1,
                  padding=0, bias=False)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.conv3 = nn.Conv2d(128, 1, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.context = context
        self._init_weights()
    def _init_weights(self):
      def normal_init(m, mean, stddev, truncated=False):
        """
        weight initalizer: truncated normal and random normal.
        """
        # x is a parameter
        if truncated:
          m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
        else:
          m.weight.data.normal_(mean, stddev)
          #m.bias.data.zero_()
      normal_init(self.conv1, 0, 0.01)
      normal_init(self.conv2, 0, 0.01)
      normal_init(self.conv3, 0, 0.01)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        if self.context:
          feat = F.avg_pool2d(x, (x.size(2), x.size(3)))
          x = self.conv3(x)
          return F.sigmoid(x),feat
        else:
          x = self.conv3(x)
          return F.sigmoid(x)

class netD2(nn.Module):
    def __init__(self,context=False):
        super(netD2, self).__init__()
        self.conv1 = conv3x3(512, 512, stride=2)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = conv3x3(512, 128, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = conv3x3(128, 128, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128,2)
        self.context = context
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    def forward(self, x):
        x = F.dropout(F.relu(self.bn1(self.conv1(x))),training=self.training)
        x = F.dropout(F.relu(self.bn2(self.conv2(x))),training=self.training)
        x = F.dropout(F.relu(self.bn3(self.conv3(x))),training=self.training)
        x = F.avg_pool2d(x,(x.size(2),x.size(3)))
        x = x.view(-1,128)
        if self.context:
          feat = x
        x = self.fc(x)
        if self.context:
          return x,feat
        else:
          return x


class netD3(nn.Module):
    def __init__(self,context=False):
        super(netD3, self).__init__()
        self.conv1 = conv3x3(512, 512, stride=2)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = conv3x3(512, 128, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = conv3x3(128, 128, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128,2)
        self.context = context
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    def forward(self, x):
        x = F.dropout(F.relu(self.bn1(self.conv1(x))),training=self.training)
        x = F.dropout(F.relu(self.bn2(self.conv2(x))),training=self.training)
        x = F.dropout(F.relu(self.bn3(self.conv3(x))),training=self.training)
        x = F.avg_pool2d(x,(x.size(2),x.size(3)))
        x = x.view(-1,128)
        if self.context:
          feat = x
        x = self.fc(x)
        if self.context:
          return x,feat
        else:
          return x

class netD_dc(nn.Module):
    def __init__(self):
        super(netD_dc, self).__init__()
        self.fc1 = nn.Linear(2048,100)
        self.bn1 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100,100)
        self.bn2 = nn.BatchNorm1d(100)
        self.fc3 = nn.Linear(100,2)
    def forward(self, x):
        x = F.dropout(F.relu(self.bn1(self.fc1(x))),training=self.training)
        x = F.dropout(F.relu(self.bn2(self.fc2(x))),training=self.training)
        x = self.fc3(x)
        return x

class vgg16_meaa(_fasterRCNN):
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

    self.RCNN_base1 = nn.Sequential(*list(vgg.features._modules.values())[:14])
    self.RCNN_base2 = nn.Sequential(*list(vgg.features._modules.values())[14:21])
    self.RCNN_base3 = nn.Sequential(*list(vgg.features._modules.values())[21:-1])
    #print(self.RCNN_base1)
    #print(self.RCNN_base2)
    self.netD1 = netD1()
    self.netD_forward1 = netD_forward1()
    self.netD2 = netD2()
    self.netD_forward2 = netD_forward2()
    self.netD3 = netD3()
    self.netD_forward3 = netD_forward3()

    self.netD1_res = netD1()
    self.netD_forward1_res = netD_forward1()
    self.netD2_res = netD2()
    self.netD_forward2_res = netD_forward2()
    self.netD3_res = netD3()
    self.netD_forward3_res = netD_forward3()

    self.fc2 = nn.Linear(128, 2)
    self.fc3 = nn.Linear(128, 2)
    self.fc2_res = nn.Linear(128, 2)
    self.fc3_res = nn.Linear(128, 2)
    feat_d = 4096
    feat_d += 128
    feat_d += 128
    feat_d += 128
#
    # Fix the layers before conv3:
    self.netD_inst = netD_inst(fc_size = feat_d)
    self.netD_inst_res = netD_inst(fc_size = feat_d)
    # Fix the layers before conv3:
    for layer in range(10):
      for p in self.RCNN_base1[layer].parameters(): p.requires_grad = False

    # self.RCNN_base = _RCNN_base(vgg.features, self.classes, self.dout_base_model)

    self.RCNN_top = vgg.classifier

    # not using the last maxpool layer
    self.RCNN_cls_score = nn.Linear(4096, self.n_classes)

    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(4096, 4)
    else:
      self.RCNN_bbox_pred = nn.Linear(4096, 4 * self.n_classes)

  def forward(self, im_data, im_info, gt_boxes, num_boxes, need_backprop,
            tgt_im_data, tgt_im_info, tgt_gt_boxes, tgt_num_boxes, tgt_need_backprop):

    assert need_backprop.detach()==1 and tgt_need_backprop.detach()==0
    eta=1
    batch_size = im_data.size(0)
    im_info = im_info.data     #(size1,size2, image ratio(new image / source image) )
    gt_boxes = gt_boxes.data
    num_boxes = num_boxes.data
    need_backprop=need_backprop.data

    '''----------stage 1-----------------------'''
    # feed image data to base model to obtain base feature map
    base_feat_rgb_1 = self.RCNN_base1(im_data)

    domain_p1 = self.netD1(grad_reverse(base_feat_rgb_1, lambd=eta))
    domain_p1_en = prob2entropy2(domain_p1)
    feat1 = base_feat_rgb_1 * domain_p1_en

    feat1 = self.netD_forward1(feat1.detach())

    feat1_p = F.softmax(feat1, 1)
    feat1_en = prob2entropy(feat1_p)
    feat1 = feat1 * feat1_en

    red = im_data[:, 0:1, :, :]
    red = torch.cat((red, im_data[:, 0:1, :, :]), 1)
    red = torch.cat((red, im_data[:, 0:1, :, :]), 1)
    green = im_data[:, 1:2, :, :]
    green = torch.cat((green, im_data[:, 1:2, :, :]), 1)
    green = torch.cat((green, im_data[:, 1:2, :, :]), 1)
    blue = im_data[:, 2:3, :, :]
    blue = torch.cat((blue, im_data[:, 2:3, :, :]), 1)
    blue = torch.cat((blue, im_data[:, 2:3, :, :]), 1)

    base_feat_red_1 = self.RCNN_base1(red)
    base_feat_green_1 = self.RCNN_base1(green)
    base_feat_blue_1 = self.RCNN_base1(blue)
    b_1, c_1, h_1, w_1 = base_feat_red_1.size()
    Rflat_1 = torch.reshape(base_feat_red_1, (b_1, 1, c_1, h_1*w_1))
    Gflat_1 = torch.reshape(base_feat_green_1, (b_1, 1, c_1, h_1*w_1))
    Bflat_1 = torch.reshape(base_feat_blue_1, (b_1, 1, c_1, h_1*w_1))
    stack_tensor_1 = torch.cat((Rflat_1, Gflat_1, Bflat_1), dim=1)
    max_tensor_1, _ = torch.max(stack_tensor_1, 1)
    min_tensor_1, _ = torch.min(stack_tensor_1, 1)
    mul_max_feat_1 = torch.reshape(max_tensor_1, (b_1, c_1, h_1, w_1))
    mul_min_feat_1 = torch.reshape(min_tensor_1, (b_1, c_1, h_1, w_1))
    max_flat_1 = torch.reshape(mul_max_feat_1, (b_1, 1, c_1, h_1*w_1))
    min_flat_1 = torch.reshape(mul_min_feat_1, (b_1, 1, c_1, h_1*w_1))
    res_tensor_1 = max_flat_1 - min_flat_1
    res_feat_1 = torch.reshape(res_tensor_1, (b_1, c_1, h_1, w_1))

    domain_p1_res = self.netD1_res(grad_reverse(res_feat_1, lambd=eta))
    domain_p1_en_res = prob2entropy2(domain_p1_res)
    feat1_res = res_feat_1 * domain_p1_en_res

    feat1_res = self.netD_forward1_res(feat1_res.detach())

    feat1_p_res = F.softmax(feat1_res, 1)
    feat1_en_res = prob2entropy(feat1_p_res)
    feat1_res = feat1_res * feat1_en_res

    '''----------stage 2------------------------'''
    base_feat_rgb_2 = self.RCNN_base2(base_feat_rgb_1)

    domain_p2 = self.netD2(grad_reverse(base_feat_rgb_2, lambd=eta))
    feat2 = self.netD_forward2(base_feat_rgb_2.detach())
    feat2_p = self.fc2(feat2.view(-1, 128))
    feat2 = global_attention(feat2, feat2_p)

    res_feat_2 = self.RCNN_base2(res_feat_1)

    domain_p2_res = self.netD2_res(grad_reverse(res_feat_2, lambd=eta))
    feat2_res = self.netD_forward2_res(res_feat_2.detach())
    feat2_p_res = self.fc2_res(feat2_res.view(-1, 128))
    feat2_res = global_attention(feat2_res, feat2_p_res)


    '''----------stage 3------------------------'''
    base_feat_rgb = self.RCNN_base3(base_feat_rgb_2)

    domain_p3 = self.netD3(grad_reverse(base_feat_rgb, lambd=eta))
    feat3 = self.netD_forward3(base_feat_rgb.detach())
    feat3_p = self.fc3(feat3.view(-1, 128))
    feat3 = global_attention(feat3, feat3_p)

    res_feat = self.RCNN_base3(res_feat_2)

    domain_p3_res = self.netD3_res(grad_reverse(res_feat, lambd=eta))
    feat3_res = self.netD_forward3_res(res_feat.detach())
    feat3_p_res = self.fc3_res(feat3_res.view(-1, 128))
    feat3_res = global_attention(feat3_res, feat3_p_res)

    base_feat = base_feat_rgb + res_feat

    # feed base feature map tp RPN to obtain rois
    self.RCNN_rpn.train()
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
        pooled_feat_rgb = self.RCNN_roi_crop(base_feat_rgb, Variable(grid_yx).detach())
        pooled_res_feat = self.RCNN_roi_crop(res_feat, Variable(grid_yx).detach())
        if cfg.CROP_RESIZE_WITH_MAX_POOL:
            pooled_feat = F.max_pool2d(pooled_feat, 2, 2)
            pooled_feat_rgb = F.max_pool2d(pooled_feat_rgb, 2, 2)
            pooled_res_feat = F.max_pool2d(pooled_res_feat, 2, 2)
    elif cfg.POOLING_MODE == 'align':
        pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        pooled_feat_rgb = self.RCNN_roi_align(base_feat_rgb, rois.view(-1, 5))
        pooled_res_feat = self.RCNN_roi_align(res_feat, rois.view(-1, 5))
    elif cfg.POOLING_MODE == 'pool':
        pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))
        pooled_feat_rgb = self.RCNN_roi_pool(base_feat_rgb, rois.view(-1,5))
        pooled_res_feat = self.RCNN_roi_pool(res_feat, rois.view(-1,5))

    # feed pooled features to top model
    pooled_feat = self._head_to_tail(pooled_feat)
    pooled_feat_rgb = self._head_to_tail(pooled_feat_rgb)
    pooled_res_feat = self._head_to_tail(pooled_res_feat)

    feat1 = feat1.view(1, -1).repeat(pooled_feat_rgb.size(0), 1)
    pooled_feat_rgb = torch.cat((feat1, pooled_feat_rgb), 1)
    feat2 = feat2.view(1, -1).repeat(pooled_feat_rgb.size(0), 1)
    pooled_feat_rgb = torch.cat((feat2, pooled_feat_rgb), 1)
    feat3 = feat3.view(1, -1).repeat(pooled_feat_rgb.size(0), 1)
    pooled_feat_rgb = torch.cat((feat3, pooled_feat_rgb), 1)

    #---------------------------------------------------------------
    d_inst = self.netD_inst(grad_reverse(pooled_feat_rgb, lambd=eta))

    feat1_res = feat1_res.view(1, -1).repeat(pooled_res_feat.size(0), 1)
    pooled_res_feat = torch.cat((feat1_res, pooled_res_feat), 1)
    feat2_res = feat2_res.view(1, -1).repeat(pooled_res_feat.size(0), 1)
    pooled_res_feat = torch.cat((feat2_res, pooled_res_feat), 1)
    feat3_res = feat3_res.view(1, -1).repeat(pooled_res_feat.size(0), 1)
    pooled_res_feat = torch.cat((feat3_res, pooled_res_feat), 1)

    #---------------------------------------------------------------
    d_inst_res = self.netD_inst_res(grad_reverse(pooled_res_feat, lambd=eta))

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

    """ =================== for target =========================="""

    tgt_batch_size = tgt_im_data.size(0)
    tgt_im_info = tgt_im_info.data  # (size1,size2, image ratio(new image / source image) )
    tgt_gt_boxes = tgt_gt_boxes.data
    tgt_num_boxes = tgt_num_boxes.data
    tgt_need_backprop = tgt_need_backprop.data

    '''------------stage 1-------------------'''
    # feed image data to base model to obtain base feature map
    tgt_base_feat_rgb_1 = self.RCNN_base1(tgt_im_data)

    tgt_domain_p1 = self.netD1(grad_reverse(tgt_base_feat_rgb_1, lambd=eta))
    tgt_domain_p1_en = prob2entropy2(tgt_domain_p1)
    tgt_feat1 = tgt_base_feat_rgb_1 * tgt_domain_p1_en

    tgt_feat1 = self.netD_forward1(tgt_feat1.detach())

    tgt_feat1_p = F.softmax(tgt_feat1, 1)
    tgt_feat1_en = prob2entropy(tgt_feat1_p)
    tgt_feat1 = tgt_feat1 * tgt_feat1_en

    # feed image data to base model to obtain base feature map
    tgt_red = tgt_im_data[:, 0:1, :, :]
    tgt_red = torch.cat((tgt_red, tgt_im_data[:, 0:1, :, :]), 1)
    tgt_red = torch.cat((tgt_red, tgt_im_data[:, 0:1, :, :]), 1)
    tgt_green = tgt_im_data[:, 1:2, :, :]
    tgt_green = torch.cat((tgt_green, tgt_im_data[:, 1:2, :, :]), 1)
    tgt_green = torch.cat((tgt_green, tgt_im_data[:, 1:2, :, :]), 1)
    tgt_blue = tgt_im_data[:, 2:3, :, :]
    tgt_blue = torch.cat((tgt_blue, tgt_im_data[:, 2:3, :, :]), 1)
    tgt_blue = torch.cat((tgt_blue, tgt_im_data[:, 2:3, :, :]), 1)
    tgt_base_feat_red_1 = self.RCNN_base1(tgt_red)
    tgt_base_feat_green_1 = self.RCNN_base1(tgt_green)
    tgt_base_feat_blue_1 = self.RCNN_base1(tgt_blue)
    b_1, c_1, h_1, w_1 = tgt_base_feat_red_1.size()
    tgt_Rflat_1 = torch.reshape(tgt_base_feat_red_1, (b_1, 1, c_1, h_1*w_1))
    tgt_Gflat_1 = torch.reshape(tgt_base_feat_green_1, (b_1, 1, c_1, h_1*w_1))
    tgt_Bflat_1 = torch.reshape(tgt_base_feat_blue_1, (b_1, 1, c_1, h_1*w_1))
    tgt_stack_tensor_1 = torch.cat((tgt_Rflat_1, tgt_Gflat_1, tgt_Bflat_1), dim=1)
    tgt_max_tensor_1, _ = torch.max(tgt_stack_tensor_1, 1)
    tgt_min_tensor_1, _ = torch.min(tgt_stack_tensor_1, 1)
    tgt_mul_max_feat_1 = torch.reshape(tgt_max_tensor_1, (b_1, c_1, h_1, w_1))
    tgt_mul_min_feat_1 = torch.reshape(tgt_min_tensor_1, (b_1, c_1, h_1, w_1))
    tgt_max_flat_1 = torch.reshape(tgt_mul_max_feat_1, (b_1, 1, c_1, h_1*w_1))
    tgt_min_flat_1 = torch.reshape(tgt_mul_min_feat_1, (b_1, 1, c_1, h_1*w_1))
    tgt_res_tensor_1 = tgt_max_flat_1 - tgt_min_flat_1
    tgt_res_feat_1 = torch.reshape(tgt_res_tensor_1, (b_1, c_1, h_1, w_1))

    tgt_domain_p1_res = self.netD1_res(grad_reverse(tgt_res_feat_1, lambd=eta))
    tgt_domain_p1_en_res = prob2entropy2(tgt_domain_p1_res)
    tgt_feat1_res = tgt_res_feat_1 * tgt_domain_p1_en_res

    tgt_feat1_res = self.netD_forward1_res(tgt_feat1_res.detach())

    tgt_feat1_p_res = F.softmax(tgt_feat1_res, 1)
    tgt_feat1_en_res = prob2entropy(tgt_feat1_p_res)
    tgt_feat1_res = tgt_feat1_res * tgt_feat1_en_res

    '''----------stage 2------------------------'''
    tgt_base_feat_rgb_2 = self.RCNN_base2(tgt_base_feat_rgb_1)

    tgt_domain_p2 = self.netD2(grad_reverse(tgt_base_feat_rgb_2, lambd=eta))
    tgt_feat2 = self.netD_forward2(tgt_base_feat_rgb_2.detach())
    tgt_feat2_p = self.fc2(tgt_feat2.view(-1, 128))
    tgt_feat2 = global_attention(tgt_feat2, tgt_feat2_p)

    tgt_res_feat_2 = self.RCNN_base2(tgt_res_feat_1)

    tgt_domain_p2_res = self.netD2_res(grad_reverse(tgt_res_feat_2, lambd=eta))
    tgt_feat2_res = self.netD_forward2_res(tgt_res_feat_2.detach())
    tgt_feat2_p_res = self.fc2_res(tgt_feat2_res.view(-1, 128))
    tgt_feat2_res = global_attention(tgt_feat2_res, feat2_p_res)

    '''----------stage 3------------------------'''
    tgt_base_feat_rgb = self.RCNN_base3(tgt_base_feat_rgb_2)

    tgt_domain_p3 = self.netD3(grad_reverse(tgt_base_feat_rgb, lambd=eta))
    tgt_feat3 = self.netD_forward3(tgt_base_feat_rgb.detach())
    tgt_feat3_p = self.fc3(tgt_feat3.view(-1, 128))
    tgt_feat3 = global_attention(tgt_feat3, tgt_feat3_p)

    tgt_res_feat = self.RCNN_base3(tgt_res_feat_2)

    tgt_domain_p3_res = self.netD3_res(grad_reverse(tgt_res_feat, lambd=eta))
    tgt_feat3_res = self.netD_forward3_res(tgt_res_feat.detach())
    tgt_feat3_p_res = self.fc3_res(tgt_feat3_res.view(-1, 128))
    tgt_feat3_res = global_attention(tgt_feat3_res, tgt_feat3_p_res)

    tgt_base_feat = tgt_base_feat_rgb + tgt_res_feat



    # feed base feature map tp RPN to obtain rois
    self.RCNN_rpn.eval()
    tgt_rois, tgt_rpn_loss_cls, tgt_rpn_loss_bbox = \
        self.RCNN_rpn(tgt_base_feat, tgt_im_info, tgt_gt_boxes, tgt_num_boxes)

    # if it is training phrase, then use ground trubut bboxes for refining

    tgt_rois_label = None
    tgt_rois_target = None
    tgt_rois_inside_ws = None
    tgt_rois_outside_ws = None
    tgt_rpn_loss_cls = 0
    tgt_rpn_loss_bbox = 0

    tgt_rois = Variable(tgt_rois)
    # do roi pooling based on predicted rois

    if cfg.POOLING_MODE == 'crop':
        # pdb.set_trace()
        # pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
        tgt_grid_xy = _affine_grid_gen(tgt_rois.view(-1, 5), tgt_base_feat.size()[2:], self.grid_size)
        tgt_grid_yx = torch.stack([tgt_grid_xy.data[:, :, :, 1], tgt_grid_xy.data[:, :, :, 0]], 3).contiguous()
        tgt_pooled_feat = self.RCNN_roi_crop(tgt_base_feat, Variable(tgt_grid_yx).detach())
        tgt_pooled_feat_rgb = self.RCNN_roi_crop(tgt_base_feat_rgb, Variable(tgt_grid_yx).detach())
        tgt_pooled_res_feat = self.RCNN_roi_crop(tgt_res_feat, Variable(tgt_grid_yx).detach())
        if cfg.CROP_RESIZE_WITH_MAX_POOL:
            tgt_pooled_feat = F.max_pool2d(tgt_base_feat, 2, 2)
            tgt_pooled_feat_rgb = F.max_pool2d(tgt_base_feat_rgb, 2, 2)
            tgt_pooled_res_feat = F.max_pool2d(tgt_res_feat, 2, 2)
    elif cfg.POOLING_MODE == 'align':
        tgt_pooled_feat = self.RCNN_roi_align(tgt_base_feat, tgt_rois.view(-1, 5))
        tgt_pooled_feat_rgb = self.RCNN_roi_align(tgt_base_feat_rgb, tgt_rois.view(-1, 5))
        tgt_pooled_res_feat = self.RCNN_roi_align(tgt_res_feat, tgt_rois.view(-1, 5))
    elif cfg.POOLING_MODE == 'pool':
        tgt_pooled_feat = self.RCNN_roi_pool(tgt_base_feat, tgt_rois.view(-1, 5))
        tgt_pooled_feat_rgb = self.RCNN_roi_pool(tgt_base_feat_rgb, tgt_rois.view(-1, 5))
        tgt_pooled_res_feat = self.RCNN_roi_pool(tgt_res_feat, tgt_rois.view(-1, 5))

    # feed pooled features to top model
    tgt_pooled_feat = self._head_to_tail(tgt_pooled_feat)
    tgt_pooled_feat_rgb = self._head_to_tail(tgt_pooled_feat_rgb)
    tgt_pooled_res_feat = self._head_to_tail(tgt_pooled_res_feat)

    tgt_feat1 = tgt_feat1.view(1, -1).repeat(tgt_pooled_feat_rgb.size(0), 1)
    tgt_pooled_feat_rgb = torch.cat((tgt_feat1, tgt_pooled_feat_rgb), 1)
    tgt_feat2 = tgt_feat2.view(1, -1).repeat(tgt_pooled_feat_rgb.size(0), 1)
    tgt_pooled_feat_rgb = torch.cat((tgt_feat2, tgt_pooled_feat_rgb), 1)
    tgt_feat3 = tgt_feat3.view(1, -1).repeat(tgt_pooled_feat_rgb.size(0), 1)
    tgt_pooled_feat_rgb = torch.cat((tgt_feat3, tgt_pooled_feat_rgb), 1)

    #---------------------------------------------------------------
    tgt_d_inst = self.netD_inst(grad_reverse(tgt_pooled_feat_rgb, lambd=eta))

    tgt_feat1_res = tgt_feat1_res.view(1, -1).repeat(tgt_pooled_res_feat.size(0), 1)
    tgt_pooled_res_feat = torch.cat((tgt_feat1_res, tgt_pooled_res_feat), 1)
    tgt_feat2_res = tgt_feat2_res.view(1, -1).repeat(tgt_pooled_res_feat.size(0), 1)
    tgt_pooled_res_feat = torch.cat((tgt_feat2_res, tgt_pooled_res_feat), 1)
    tgt_feat3_res = tgt_feat3_res.view(1, -1).repeat(tgt_pooled_res_feat.size(0), 1)
    tgt_pooled_res_feat = torch.cat((tgt_feat3_res, tgt_pooled_res_feat), 1)

    #---------------------------------------------------------------
    tgt_d_inst_res = self.netD_inst_res(grad_reverse(tgt_pooled_res_feat, lambd=eta))



    """  DA loss feat  """
    domain_s2 = domain_s3 = Variable(torch.ones(domain_p2.size(0)).long().cuda())
    domain_s_p = Variable(torch.ones(d_inst.size(0)).long().cuda())
    # k=1th loss
    dloss_s1 = 0.5 * torch.mean(domain_p1 ** 2)
    # k=2nd loss
    dloss_s2 = 0.5 * CrossEntropy(domain_p2, domain_s2) * 0.15
    # k = 3rd loss
    FL = FocalLoss(class_num=2, gamma=5)
    dloss_s3 = 0.5 * FL(domain_p3, domain_s3)
    # instance alignment loss
    dloss_s_p = 0.5 * FL(d_inst, domain_s_p)
    # new losses
    feat1_s_p = 0.5 * torch.mean(feat1 ** 2)
    feat2_s_p = 0.5 * torch.mean(feat2 ** 2)
    feat3_s_p = 0.5 * torch.mean(feat3 ** 2)

    domain_s2_res = domain_s3_res = Variable(torch.ones(domain_p2_res.size(0)).long().cuda())
    domain_s_p_res = Variable(torch.ones(d_inst_res.size(0)).long().cuda())
    # k=1th loss
    dloss_s1_res = 0.5 * torch.mean(domain_p1_res ** 2)
    # k=2nd loss
    dloss_s2_res = 0.5 * CrossEntropy(domain_p2_res, domain_s2_res) * 0.15
    # k = 3rd loss
    dloss_s3_res = 0.5 * FL(domain_p3_res, domain_s3_res)
    # instance alignment loss
    dloss_s_p_res = 0.5 * FL(d_inst_res, domain_s_p_res)
    # new losses
    feat1_s_p_res = 0.5 * torch.mean(feat1_res ** 2)
    feat2_s_p_res = 0.5 * torch.mean(feat2_res ** 2)
    feat3_s_p_res = 0.5 * torch.mean(feat3_res ** 2)

    DA_cst_loss=self.consistency_loss(torch.mean(d_inst,0).view(1,2),domain_p3.detach())

    DA_cst_loss_res=self.consistency_loss(torch.mean(d_inst_res,0).view(1,2),domain_p3_res.detach())

    """  ************** taget loss ****************  """
    tgt_domain_s2 = tgt_domain_s3 = Variable(torch.zeros(tgt_domain_p2.size(0)).long().cuda())
    tgt_domain_s_p = Variable(torch.zeros(tgt_d_inst.size(0)).long().cuda())
    # k=1th loss
    tgt_dloss_s1 = 0.5 * torch.mean(tgt_domain_p1 ** 2)
    # k=2nd loss
    tgt_dloss_s2 = 0.5 * CrossEntropy(tgt_domain_p2, tgt_domain_s2) * 0.15
    # k = 3rd loss
    tgt_dloss_s3 = 0.5 * FL(tgt_domain_p3, tgt_domain_s3)
    # instance alignment loss
    tgt_dloss_s_p = 0.5 * FL(tgt_d_inst, tgt_domain_s_p)
    # new losses
    tgt_feat1_s_p = 0.5 * torch.mean(tgt_feat1 ** 2)
    tgt_feat2_s_p = 0.5 * torch.mean(tgt_feat2 ** 2)
    tgt_feat3_s_p = 0.5 * torch.mean(tgt_feat3 ** 2)

    tgt_domain_s2_res = tgt_domain_s3_res = Variable(torch.zeros(tgt_domain_p2_res.size(0)).long().cuda())
    tgt_domain_s_p_res = Variable(torch.ones(tgt_d_inst_res.size(0)).long().cuda())
    # k=1th loss
    tgt_dloss_s1_res = 0.5 * torch.mean(tgt_domain_p1_res ** 2)
    # k=2nd loss
    tgt_dloss_s2_res = 0.5 * CrossEntropy(tgt_domain_p2_res, tgt_domain_s2_res) * 0.15
    # k = 3rd loss
    tgt_dloss_s3_res = 0.5 * FL(tgt_domain_p3_res, tgt_domain_s3_res)
    # instance alignment loss
    tgt_dloss_s_p_res = 0.5 * FL(tgt_d_inst_res, tgt_domain_s_p_res)
    # new losses
    tgt_feat1_s_p_res = 0.5 * torch.mean(tgt_feat1_res ** 2)
    tgt_feat2_s_p_res = 0.5 * torch.mean(tgt_feat2_res ** 2)
    tgt_feat3_s_p_res = 0.5 * torch.mean(tgt_feat3_res ** 2)

    tgt_DA_cst_loss=self.consistency_loss(torch.mean(tgt_d_inst,0).view(1,2),tgt_domain_p3.detach())

    tgt_DA_cst_loss_res=self.consistency_loss(torch.mean(tgt_d_inst_res,0).view(1,2),tgt_domain_p3_res.detach())


    mDA_cst_loss=self.consistency_loss(torch.mean(d_inst_res,0).view(1,2),domain_p3.detach())
    tgt_mDA_cst_loss = self.consistency_loss(torch.mean(tgt_d_inst_res,0).view(1,2), tgt_domain_p3.detach())


    dloss = (dloss_s1 + dloss_s2 + tgt_dloss_s1 + tgt_dloss_s2 + dloss_s3 + tgt_dloss_s3 + dloss_s_p + tgt_dloss_s_p)
    dloss_res = (dloss_s1_res + dloss_s2_res + tgt_dloss_s1_res + tgt_dloss_s2_res + dloss_s3_res + tgt_dloss_s3_res + dloss_s_p_res + tgt_dloss_s_p_res)
    featloss = (feat1_s_p + feat2_s_p + feat3_s_p + tgt_feat1_s_p + tgt_feat2_s_p + tgt_feat3_s_p)
    featloss_res = (feat1_s_p_res + feat2_s_p_res + feat3_s_p_res + tgt_feat1_s_p_res + tgt_feat2_s_p_res + tgt_feat3_s_p_res)
    cst_loss = DA_cst_loss + tgt_DA_cst_loss
    cst_loss_res = DA_cst_loss_res + tgt_DA_cst_loss_res


    return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label,\
           dloss, dloss_res, featloss, featloss_res, cst_loss, cst_loss_res, mDA_cst_loss, tgt_mDA_cst_loss

  def _head_to_tail(self, pool5):

    pool5_flat = pool5.view(pool5.size(0), -1)
    fc7 = self.RCNN_top(pool5_flat)

    return fc7
