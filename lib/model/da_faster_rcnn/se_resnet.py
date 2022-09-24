from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.utils.config import cfg
from model.da_faster_rcnn.faster_rcnn import _fasterRCNN
from model.da_faster_rcnn.DA import _InstanceDA
import torch.nn as nn
#from torch.hub import load_state_dict_from_url
from torchvision.models import ResNet
from model.da_faster_rcnn.se_module import SELayer
from model.da_faster_rcnn.resnet import resnet18, resnet34, resnet50, resnet101, resnet152

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
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
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
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def se_resnet18(num_classes=1_000):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet34(num_classes=1_000):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet50(num_classes=1_000, pretrained=False):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    #if pretrained:
    #    model.load_state_dict(load_state_dict_from_url(
    #        "https://github.com/moskomule/senet.pytorch/releases/download/archive/seresnet50-60a8950a85b2b.pkl"))
    return model


def se_resnet101(num_classes=1_000):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 4, 23, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet152(num_classes=1_000):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 8, 36, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


class CifarSEBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, reduction=16):
        super(CifarSEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        if inplanes != planes:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm2d(planes))
        else:
            self.downsample = lambda x: x
        self.stride = stride

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        out += residual
        out = self.relu(out)

        return out


class CifarSEResNet(nn.Module):
    def __init__(self, block, n_size, num_classes=10, reduction=16):
        super(CifarSEResNet, self).__init__()
        self.inplane = 16
        self.conv1 = nn.Conv2d(
            3, self.inplane, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(
            block, 16, blocks=n_size, stride=1, reduction=reduction)
        self.layer2 = self._make_layer(
            block, 32, blocks=n_size, stride=2, reduction=reduction)
        self.layer3 = self._make_layer(
            block, 64, blocks=n_size, stride=2, reduction=reduction)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride, reduction):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplane, planes, stride, reduction))
            self.inplane = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class CifarSEPreActResNet(CifarSEResNet):
    def __init__(self, block, n_size, num_classes=10, reduction=16):
        super(CifarSEPreActResNet, self).__init__(
            block, n_size, num_classes, reduction)
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.initialize()

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn1(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)


def se_resnet20(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = CifarSEResNet(CifarSEBasicBlock, 3, **kwargs)
    return model


def se_resnet32(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = CifarSEResNet(CifarSEBasicBlock, 5, **kwargs)
    return model


def se_resnet56(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = CifarSEResNet(CifarSEBasicBlock, 9, **kwargs)
    return model


def se_preactresnet20(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = CifarSEPreActResNet(CifarSEBasicBlock, 3, **kwargs)
    return model


def se_preactresnet32(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = CifarSEPreActResNet(CifarSEBasicBlock, 5, **kwargs)
    return model


def se_preactresnet56(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = CifarSEPreActResNet(CifarSEBasicBlock, 9, **kwargs)
    return model

class se_resnetDA(_fasterRCNN):
  def __init__(self, classes, num_layers=101, pretrained=False, class_agnostic=False):
    self.model_path = 'data/pretrained_model/resnet101_caffe.pth'
    self.dout_base_model = 1024
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic

    _fasterRCNN.__init__(self, classes, class_agnostic)
    self.RCNN_instanceDA = _InstanceDA(resnet=True)

  def _init_modules(self):
    resnet = resnet101()

    if self.pretrained == True:
      print("Loading pretrained weights from %s" %(self.model_path))
      state_dict = torch.load(self.model_path)
      resnet.load_state_dict({k:v for k,v in state_dict.items() if k in resnet.state_dict()})

    # Build resnet.
    self.RCNN_base = nn.Sequential(resnet.conv1, resnet.bn1,resnet.relu,
      resnet.maxpool,resnet.layer1,resnet.layer2,resnet.layer3)

    self.RCNN_top = nn.Sequential(resnet.layer4)

    self.RCNN_cls_score = nn.Linear(2048, self.n_classes)
    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(2048, 4)
    else:
      self.RCNN_bbox_pred = nn.Linear(2048, 4 * self.n_classes)

    # Fix blocks
    '''for p in self.RCNN_base[0].parameters(): p.requires_grad=False
    for p in self.RCNN_base[1].parameters(): p.requires_grad=False

    assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
    if cfg.RESNET.FIXED_BLOCKS >= 3:
      for p in self.RCNN_base[6].parameters(): p.requires_grad=False
    if cfg.RESNET.FIXED_BLOCKS >= 2:
      for p in self.RCNN_base[5].parameters(): p.requires_grad=False
    if cfg.RESNET.FIXED_BLOCKS >= 1:
      for p in self.RCNN_base[4].parameters(): p.requires_grad=False'''

    def set_bn_fix(m):
      classname = m.__class__.__name__
      if classname.find('BatchNorm') != -1:
        for p in m.parameters(): p.requires_grad=False

    self.RCNN_base.apply(set_bn_fix)
    self.RCNN_top.apply(set_bn_fix)

  '''def train(self, mode=True):
    #print(self.RCNN_base)
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
      self.RCNN_top.apply(set_bn_eval)'''

  def forward(self, im_data, im_info, gt_boxes, num_boxes, need_backprop):


    batch_size = im_data.size(0)
    im_info = im_info.data     #(size1,size2, image ratio(new image / source image) )
    gt_boxes = gt_boxes.data
    num_boxes = num_boxes.data
    need_backprop=need_backprop.data

    # feed image data to base model to obtain base feature map
    base_feat = self.RCNN_base(im_data)

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
        if cfg.CROP_RESIZE_WITH_MAX_POOL:
            pooled_feat = F.max_pool2d(pooled_feat, 2, 2)
    elif cfg.POOLING_MODE == 'align':
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
    fc7 = self.RCNN_top(pool5).mean(3).mean(2)
    return fc7

  def getbackbone(self):
      return self.RCNN_base
