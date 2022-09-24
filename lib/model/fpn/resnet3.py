from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.utils.config import cfg
from model.fpn.fpn import _FPN

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo
import pdb
import torchvision.utils as vutils
from model.utils.config import cfg
from model.rpn.rpn_fpn import _RPN_FPN
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg2
from model.rpn.proposal_target_layer import _ProposalTargetLayer
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta
import time
import pdb

from model.da_faster_rcnn.DA import _ImageDA
from model.da_faster_rcnn.DA import _InstanceDA

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
       'resnet152']


model_urls = {
  'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
  'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
  'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
  'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
  'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
           padding=1, bias=False)


class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(BasicBlock, self).__init__()
    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = nn.BatchNorm2d(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = nn.BatchNorm2d(planes)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(Bottleneck, self).__init__()
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                 padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(planes * 4)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class ResNet(nn.Module):
  def __init__(self, block, layers, num_classes=1000):
    self.inplanes = 64
    super(ResNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                 bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True) # change
    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
    self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
    self.avgpool = nn.AvgPool2d(7)
    self.fc = nn.Linear(512 * block.expansion, num_classes)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
              kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)

    return x


def resnet18(pretrained=False):
  """Constructs a ResNet-18 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(BasicBlock, [2, 2, 2, 2])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
  return model


def resnet34(pretrained=False):
  """Constructs a ResNet-34 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(BasicBlock, [3, 4, 6, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
  return model


def resnet50(pretrained=False):
  """Constructs a ResNet-50 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 4, 6, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
  return model


def resnet101(pretrained=False):
  """Constructs a ResNet-101 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 4, 23, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
  return model


def resnet152(pretrained=False):
  """Constructs a ResNet-152 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 8, 36, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
  return model

class resnet_fpn(_FPN):
  def __init__(self, classes, num_layers=101, pretrained=False, class_agnostic=False):
    self.model_path = 'data/pretrained_model/resnet101_caffe.pth'
    self.dout_base_model = 256
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic

    _FPN.__init__(self, classes, class_agnostic)

  def _init_modules(self):
    resnet = resnet101()

    if self.pretrained == True:
      print("Loading pretrained weights from %s" %(self.model_path))
      state_dict = torch.load(self.model_path)
      resnet.load_state_dict({k:v for k,v in state_dict.items() if k in resnet.state_dict()})

    self.RCNN_layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
    self.RCNN_layer1 = nn.Sequential(resnet.layer1)
    self.RCNN_layer2 = nn.Sequential(resnet.layer2)
    self.RCNN_layer3 = nn.Sequential(resnet.layer3)
    self.RCNN_layer4 = nn.Sequential(resnet.layer4)

    # Top layer
    self.RCNN_toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # reduce channel

    # Smooth layers
    self.RCNN_smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
    self.RCNN_smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
    self.RCNN_smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

    # Lateral layers
    self.RCNN_latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
    self.RCNN_latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
    self.RCNN_latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)

    # ROI Pool feature downsampling
    self.RCNN_roi_feat_ds = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)

    self.RCNN_top = nn.Sequential(
      nn.Conv2d(256, 1024, kernel_size=cfg.POOLING_SIZE, stride=cfg.POOLING_SIZE, padding=0),
      nn.ReLU(True),
      nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0),
      nn.ReLU(True)
      )

    self.RCNN_imageDA = _ImageDA(1024)
    self.RCNN_cls_score = nn.Linear(1024, self.n_classes)
    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(1024, 4)
    else:
      self.RCNN_bbox_pred = nn.Linear(1024, 4 * self.n_classes)

    # Fix blocks
    for p in self.RCNN_layer0[0].parameters(): p.requires_grad=False
    for p in self.RCNN_layer0[1].parameters(): p.requires_grad=False

    assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
    if cfg.RESNET.FIXED_BLOCKS >= 3:
      for p in self.RCNN_layer3.parameters(): p.requires_grad=False
    if cfg.RESNET.FIXED_BLOCKS >= 2:
      for p in self.RCNN_layer2.parameters(): p.requires_grad=False
    if cfg.RESNET.FIXED_BLOCKS >= 1:
      for p in self.RCNN_layer1.parameters(): p.requires_grad=False

    def set_bn_fix(m):
      classname = m.__class__.__name__
      if classname.find('BatchNorm') != -1:
        for p in m.parameters(): p.requires_grad=False

    self.RCNN_layer0.apply(set_bn_fix)
    self.RCNN_layer1.apply(set_bn_fix)
    self.RCNN_layer2.apply(set_bn_fix)
    self.RCNN_layer3.apply(set_bn_fix)
    self.RCNN_layer4.apply(set_bn_fix)

  def train(self, mode=True):
    # Override train so that the training mode is set as we want
    nn.Module.train(self, mode)
    if mode:
      # Set fixed blocks to be in eval mode
      self.RCNN_layer0.eval()
      self.RCNN_layer1.eval()
      self.RCNN_layer2.train()
      self.RCNN_layer3.train()
      self.RCNN_layer4.train()

      self.RCNN_smooth1.train()
      self.RCNN_smooth2.train()
      self.RCNN_smooth3.train()

      self.RCNN_latlayer1.train()
      self.RCNN_latlayer2.train()
      self.RCNN_latlayer3.train()

      self.RCNN_toplayer.train()

      def set_bn_eval(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
          m.eval()

      self.RCNN_layer0.apply(set_bn_eval)
      self.RCNN_layer1.apply(set_bn_eval)
      self.RCNN_layer2.apply(set_bn_eval)
      self.RCNN_layer3.apply(set_bn_eval)
      self.RCNN_layer4.apply(set_bn_eval)

  def forward(self, im_data, im_info, gt_boxes, num_boxes, need_backprop,
            tgt_im_data, tgt_im_info, tgt_gt_boxes, tgt_num_boxes, tgt_need_backprop):
    assert need_backprop.detach()==1 and tgt_need_backprop.detach()==0

    batch_size = im_data.size(0)
    im_info = im_info.data
    gt_boxes = gt_boxes.data
    num_boxes = num_boxes.data
    need_backprop=need_backprop.data

    # feed image data to base model to obtain base feature map
    # Bottom-up
    c1 = self.RCNN_layer0(im_data)
    c2 = self.RCNN_layer1(c1)
    c3 = self.RCNN_layer2(c2)
    c4 = self.RCNN_layer3(c3)
    c5 = self.RCNN_layer4(c4)
    # Top-down
    p5 = self.RCNN_toplayer(c5)
    p4 = self._upsample_add(p5, self.RCNN_latlayer1(c4))
    p4 = self.RCNN_smooth1(p4)
    p3 = self._upsample_add(p4, self.RCNN_latlayer2(c3))
    p3 = self.RCNN_smooth2(p3)
    p2 = self._upsample_add(p3, self.RCNN_latlayer3(c2))
    p2 = self.RCNN_smooth3(p2)

    p6 = self.maxpool2d(p5)

    rpn_feature_maps = [p2, p3, p4, p5, p6]
    mrcnn_feature_maps = [p2, p3, p4, p5]

    self.RCNN_rpn.train()
    rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(rpn_feature_maps, im_info, gt_boxes, num_boxes)

    # if it is training phrase, then use ground trubut bboxes for refining
    if self.training:
        roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
        rois, rois_label, gt_assign, rois_target, rois_inside_ws, rois_outside_ws = roi_data

        ## NOTE: additionally, normalize proposals to range [0, 1],
        #        this is necessary so that the following roi pooling
        #        is correct on different feature maps
        # rois[:, :, 1::2] /= im_info[0][1]
        # rois[:, :, 2::2] /= im_info[0][0]

        rois = rois.view(-1, 5)
        rois_label = rois_label.view(-1).long()
        gt_assign = gt_assign.view(-1).long()
        pos_id = rois_label.nonzero().squeeze()
        gt_assign_pos = gt_assign[pos_id]
        rois_label_pos = rois_label[pos_id]
        rois_label_pos_ids = pos_id

        rois_pos = Variable(rois[pos_id])
        rois = Variable(rois)
        rois_label = Variable(rois_label)

        rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
        rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
        rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
    else:
        ## NOTE: additionally, normalize proposals to range [0, 1],
        #        this is necessary so that the following roi pooling
        #        is correct on different feature maps
        # rois[:, :, 1::2] /= im_info[0][1]
        # rois[:, :, 2::2] /= im_info[0][0]

        rois_label = None
        gt_assign = None
        rois_target = None
        rois_inside_ws = None
        rois_outside_ws = None
        rpn_loss_cls = 0
        rpn_loss_bbox = 0
        rois = rois.view(-1, 5)
        pos_id = torch.arange(0, rois.size(0)).long().type_as(rois).long()
        rois_label_pos_ids = pos_id
        rois_pos = Variable(rois[pos_id])
        rois = Variable(rois)

    # pooling features based on rois, output 14x14 map
    roi_pool_feat = self._PyramidRoI_Feat(mrcnn_feature_maps, rois, im_info)

    # feed pooled features to top model
    pooled_feat = self._head_to_tail(roi_pool_feat)


    # compute bbox offset
    bbox_pred = self.RCNN_bbox_pred(pooled_feat)
    if self.training and not self.class_agnostic:
        # select the corresponding columns according to roi labels
        bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
        bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.long().view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
        bbox_pred = bbox_pred_select.squeeze(1)

    # compute object classification probability
    cls_score = self.RCNN_cls_score(pooled_feat)
    cls_prob = F.softmax(cls_score)

    RCNN_loss_cls = 0
    RCNN_loss_bbox = 0

    if self.training:
        # loss (cross entropy) for object classification
        RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)
        # loss (l1-norm) for bounding box regression
        RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

    rois = rois.view(batch_size, -1, rois.size(1))
    cls_prob = cls_prob.view(batch_size, -1, cls_prob.size(1))
    bbox_pred = bbox_pred.view(batch_size, -1, bbox_pred.size(1))

    if self.training:
        rois_label = rois_label.view(batch_size, -1)

    """ =================== for target =========================="""

    tgt_batch_size = tgt_im_data.size(0)
    tgt_im_info = tgt_im_info.data
    tgt_gt_boxes = tgt_gt_boxes.data
    tgt_num_boxes = tgt_num_boxes.data
    tgt_need_backprop=tgt_need_backprop.data

    # feed image data to base model to obtain base feature map
    # Bottom-up
    tgt_c1 = self.RCNN_layer0(tgt_im_data)
    tgt_c2 = self.RCNN_layer1(tgt_c1)
    tgt_c3 = self.RCNN_layer2(tgt_c2)
    tgt_c4 = self.RCNN_layer3(tgt_c3)
    tgt_c5 = self.RCNN_layer4(tgt_c4)
    # Top-down
    tgt_p5 = self.RCNN_toplayer(tgt_c5)
    tgt_p4 = self._upsample_add(tgt_p5, self.RCNN_latlayer1(tgt_c4))
    tgt_p4 = self.RCNN_smooth1(tgt_p4)
    tgt_p3 = self._upsample_add(tgt_p4, self.RCNN_latlayer2(tgt_c3))
    tgt_p3 = self.RCNN_smooth2(tgt_p3)
    tgt_p2 = self._upsample_add(tgt_p3, self.RCNN_latlayer3(tgt_c2))
    tgt_p2 = self.RCNN_smooth3(tgt_p2)

    tgt_p6 = self.maxpool2d(tgt_p5)

    tgt_rpn_feature_maps = [tgt_p2, tgt_p3, tgt_p4, tgt_p5, tgt_p6]
    tgt_mrcnn_feature_maps = [tgt_p2, tgt_p3, tgt_p4, tgt_p5]
    self.RCNN_rpn.eval()
    tgt_rois, tgt_rpn_loss_cls, tgt_rpn_loss_bbox = self.RCNN_rpn(tgt_rpn_feature_maps, tgt_im_info, tgt_gt_boxes, tgt_num_boxes)

    tgt_rois_label = None
    tgt_gt_assign = None
    tgt_rois_target = None
    tgt_rois_inside_ws = None
    tgt_rois_outside_ws = None
    tgt_rpn_loss_cls = 0
    tgt_rpn_loss_bbox = 0
    tgt_rois = tgt_rois.view(-1, 5)
    tgt_pos_id = torch.arange(0, tgt_rois.size(0)).long().type_as(tgt_rois).long()
    tgt_rois_label_pos_ids = tgt_pos_id
    tgt_rois_pos = Variable(tgt_rois[tgt_pos_id])
    tgt_rois = Variable(tgt_rois)

    # pooling features based on rois, output 14x14 map
    tgt_roi_pool_feat = self._PyramidRoI_Feat(tgt_mrcnn_feature_maps, tgt_rois, tgt_im_info)

    # feed pooled features to top model
    tgt_pooled_feat = self._head_to_tail(tgt_roi_pool_feat)


    # compute bbox offset
    tgt_bbox_pred = self.RCNN_bbox_pred(tgt_pooled_feat)

    """  DA loss   """
    # DA LOSS
    DA_img_loss_cls = 0
    DA_ins_loss_cls = 0

    tgt_DA_img_loss_cls = 0
    tgt_DA_ins_loss_cls = 0

    base_score, base_label = self.RCNN_imageDA(c4, need_backprop)

    # Image DA
    base_prob = F.log_softmax(base_score, dim=1)
    DA_img_loss_cls = F.nll_loss(base_prob, base_label)

    instance_sigmoid, same_size_label = self.RCNN_instanceDA(pooled_feat, need_backprop)
    instance_loss = nn.BCELoss()
    DA_ins_loss_cls = instance_loss(instance_sigmoid, same_size_label)

    #consistency_prob = torch.max(F.softmax(base_score, dim=1),dim=1)[0]
    consistency_prob = F.softmax(base_score, dim=1)[:,1,:,:]
    consistency_prob=torch.mean(consistency_prob)
    consistency_prob=consistency_prob.repeat(instance_sigmoid.size())

    DA_cst_loss=self.consistency_loss(instance_sigmoid,consistency_prob.detach())

    """  ************** taget loss ****************  """

    tgt_base_score, tgt_base_label = \
        self.RCNN_imageDA(tgt_c4, tgt_need_backprop)

    # Image DA
    tgt_base_prob = F.log_softmax(tgt_base_score, dim=1)
    tgt_DA_img_loss_cls = F.nll_loss(tgt_base_prob, tgt_base_label)


    tgt_instance_sigmoid, tgt_same_size_label = \
        self.RCNN_instanceDA(tgt_pooled_feat, tgt_need_backprop)
    tgt_instance_loss = nn.BCELoss()

    tgt_DA_ins_loss_cls = \
        tgt_instance_loss(tgt_instance_sigmoid, tgt_same_size_label)


    tgt_consistency_prob = F.softmax(tgt_base_score, dim=1)[:, 0, :, :]
    tgt_consistency_prob = torch.mean(tgt_consistency_prob)
    tgt_consistency_prob = tgt_consistency_prob.repeat(tgt_instance_sigmoid.size())

    tgt_DA_cst_loss = self.consistency_loss(tgt_instance_sigmoid, tgt_consistency_prob.detach())

    return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label,\
           DA_img_loss_cls,DA_ins_loss_cls,tgt_DA_img_loss_cls,tgt_DA_ins_loss_cls,DA_cst_loss,tgt_DA_cst_loss


  def _head_to_tail(self, pool5):
    block5 = self.RCNN_top(pool5)
    fc7 = block5.mean(3).mean(2)
    return fc7
