from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.utils.config import cfg
from model.faster_rcnn.faster_rcnn import _fasterRCNN

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta
from model.da_faster_rcnn.DA import _ImageDA
from model.da_faster_rcnn.DA import _InstanceDA
from model.da_faster_rcnn.DA import grad_reverse
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision
from model.roi_align.modules.roi_align import RoIAlignAvg

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
    self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
    # it is slightly better whereas slower to set stride = 1
    # self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
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

def conv3x3(in_planes, out_planes, stride=1, padding=0):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
           padding=padding, bias=False)
def conv1x1(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
           padding=0, bias=False)

class PEN(nn.Module):
    def __init__(self):
        super(PEN, self).__init__()
        self.conv1 = conv1x1(1024, 128)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = conv3x3(128, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = conv3x3(64, 64)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = conv3x3(64, 3)

    def forward(self, x):
        x = grad_reverse(x)
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = F.sigmoid(self.conv4(x))

        return x

class MR(nn.Module):
    def __init__(self,dim):
        super(MR, self).__init__()
        self.conv1 = conv1x1(dim, 256)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = conv3x3(256, 128, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = conv3x3(128, 128, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = F.relu(self.conv3(x))

        return x

class resnetDA_transmission_scl_mr(_fasterRCNN):
  def __init__(self, classes, num_layers=101, pretrained=False, class_agnostic=False):
    self.model_path = '/data/ztc/detectionModel/resnet101_caffe.pth'
    self.dout_base_model = 1024
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic

    _fasterRCNN.__init__(self, classes, class_agnostic)

  def _init_modules(self):
    resnet = resnet101()

    if self.pretrained == True:
      print("Loading pretrained weights from %s" %(self.model_path))
      state_dict = torch.load(self.model_path)
      resnet.load_state_dict({k:v for k,v in state_dict.items() if k in resnet.state_dict()})


    # Build resnet.
    self.RCNN_base = nn.Sequential(resnet.conv1, resnet.bn1,resnet.relu,
      resnet.maxpool,resnet.layer1,resnet.layer2, resnet.layer3)

    self.RCNN_roi_align_1 = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
    self.RCNN_roi_align_2 = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
    self.RCNN_top = nn.Sequential(resnet.layer4)

    self.mr1 = MR(256)
    self.mr2 = MR(512)

    feat_d = 2048
    feat_d += 128
    feat_d += 128

    self.RCNN_cls_score = nn.Linear(feat_d, self.n_classes)
    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(feat_d, 4)
    else:
      self.RCNN_bbox_pred = nn.Linear(feat_d, 4 * self.n_classes)

    self.pen = PEN()

    self.RCNN_instanceDA = _InstanceDA(resnet=True)
    # Fix blocks
    for p in self.RCNN_base[0].parameters(): p.requires_grad=False
    for p in self.RCNN_base[1].parameters(): p.requires_grad=False

    assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
    if cfg.RESNET.FIXED_BLOCKS >= 3:
      for p in self.RCNN_base[6].parameters(): p.requires_grad=False
    if cfg.RESNET.FIXED_BLOCKS >= 2:
      for p in self.RCNN_base[5].parameters(): p.requires_grad=False
    if cfg.RESNET.FIXED_BLOCKS >= 1:
      for p in self.RCNN_base[4].parameters(): p.requires_grad=False

    def set_bn_fix(m):
      classname = m.__class__.__name__
      if classname.find('BatchNorm') != -1:
        for p in m.parameters(): p.requires_grad=False

    self.RCNN_base.apply(set_bn_fix)
    self.RCNN_top.apply(set_bn_fix)

  def forward(self, im_data, im_info, gt_boxes, num_boxes):
    batch_size = im_data.size(0)

    im_info = im_info.data     #(size1,size2, image ratio(new image / source image) )
    gt_boxes = gt_boxes.data
    num_boxes = num_boxes.data

    # feed image data to base model to obtain base feature map
    base_feat_1 = self.RCNN_base[:5](im_data)
    feat1 = self.mr1(base_feat_1.detach())
    base_feat_2 = self.RCNN_base[5](base_feat_1)
    feat2 = self.mr2(base_feat_2.detach())
    base_feat = self.RCNN_base[6](base_feat_2)

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
        pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        pooled_feat_1 = self.RCNN_roi_align_1(feat1, rois.view(-1, 5))
        pooled_feat_2 = self.RCNN_roi_align_2(feat2, rois.view(-1, 5))
    elif cfg.POOLING_MODE == 'pool':
        pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))

    # feed pooled features to top model
    pooled_feat = self._head_to_tail(pooled_feat)
    pooled_feat_1 = pooled_feat_1.mean(3).mean(2).view(300,-1)
    pooled_feat_2 = pooled_feat_2.mean(3).mean(2).view(300,-1)
    pooled_feat = torch.cat((pooled_feat, pooled_feat_1), 1)
    pooled_feat = torch.cat((pooled_feat, pooled_feat_2), 1)

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

  def train(self, mode=True):
    # Override train so that the training mode is set as we want
    nn.Module.train(self, mode)
    if mode:
      # Set fixed blocks to be in eval mode
      self.RCNN_base.eval()
      self.RCNN_base[5].train()
      self.RCNN_base[6].train()

      def set_bn_eval(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
          m.eval()

      self.RCNN_base.apply(set_bn_eval)
      self.RCNN_top.apply(set_bn_eval)

  def _head_to_tail(self, pool5):
    fc7 = self.RCNN_top(pool5).mean(3).mean(2)
    return fc7
