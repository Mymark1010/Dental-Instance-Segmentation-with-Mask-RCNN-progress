from collections import OrderedDict
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.model_zoo import load_url #load link tu web
from torchvision import models
from torchvision.ops import misc 

from .utils import AnchorGenerator
from .rpn import RPNHead, RegionProposalNetwork
from .pooler import RoIAlign
from .roi_heads import RoIHeads
from .transform import Transformer


class MaskRCNN(nn.Module):
    def __init__(self, backbone, num_classes, 
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_num_samples=256, rpn_positive_fraction=0.5,
                 rpn_reg_weights=(1., 1., 1., 1.),
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000, 
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_num_samples=512, box_positive_fraction=0.25,
                 box_reg_weights=(10., 10., 5., 5.),
                 box_score_thresh=0.3, box_nms_thresh=0.6, box_num_detections=100):
        """input class MaskRCNN:
            Tham so trong RPN
                + backbone: duoc dinh nghia truoc trong nn.Module (backbone duoc su dung tinh feature trong model)
                + num_classes: int (so luong output classes cua model, bao gom ca background)
                + rpn_fg_iou_thresh: float (nguong IoU toi thieu giua anchor va gt box de anchor duoc coi la positive trong qua trinh train RPN) 
                + rpn_bg_iou_thresh: float (nguong IoU toi da giua anchor va gt box de anchor duoc coi la negative trong qua trinh train RPN)
                + rpn_num_samples: int (so luong anchors duoc lay trong moi lan train de tinh loss (posi+nega))
                + rpn_positive_fraction: float (ti le anchor positive trong tong so anchor duoc lay de train RPN)
                + rpn_reg_weights: (Tuple[float, float, float, float]) (cac trong so [wx, wy, ww, wh] duoc su dung de chuan hoa trong encode vaf decode box_ops.py)
                + rpn_pre_nms_top_n_train: int (so luong proposals duoc giu truoc khi ap dung nms + top k trong train)
                + rpn_pre_nms_top_n_test: int (so luong proposals duoc giu truoc khi ap dung nms + top k trong test)
                + rpn_post_nms_top_n_train: int (so luong proposals duoc giu sau khi ap dung nms + top k trong train)
                + rpn_post_nms_top_n_test: int (so luong proposals duoc giu sau khi ap dung nms + top k trong test)
                + rpn_nms_thresh: float (nguong ioU dung trong nms de loc cac proposals bi trung nhau trong RPN)
                
            Tham so trong RoIHead (classification + box regression Head)
                + box_fg_iou_thresh: float (nguong iou toi thieu giua proposals va gt box de proposal duoc coi la postivei trong giai doan train classification head)
                + box_bg_iou_thresh: float (nguong iou toi da giua proposals va gt box de proposal duoc coi la negative trong giai doan train classification head
                + box_num_samples: int (so luong proposals duoc lay trong train classification head de tinh loss)
                + box_positive_fraction: float (ti le positive proposal trong tong so proposal duoc lay o box_num_samples)
                + box_reg_weights: (Tuple[float, float, float, float]) (trong so duoc su dung de chuan hoa trong encode va decode box_ops.py trong classification head)
                + box_score_thresh: float (trong giai doan test, chi giu lai nhung proposals co classification score >= box_score_thresh)
                + box_nms_thresh: float (nguong iou trong giai doan test, cho nms trong classificaiton head loc final detect)
                + box_num_detections: int (so luong detection toi da tren toan bo class, trong ket qua cuoi cung)  
                """
        super().__init__()
        self.backbone = backbone
        out_channels = backbone.out_channels

        #RPN
        anchor_sizes = (16, 32, 64, 128, 256)
        anchor_ratios = (0.5, 1, 2)
        num_anchors = len(anchor_sizes) * len(anchor_ratios)
        rpn_anchor_generator = AnchorGenerator(anchor_sizes, anchor_ratios)
        rpn_head = RPNHead(out_channels, num_anchors)

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n=dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)
        self.rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_num_samples, rpn_positive_fraction, rpn_reg_weights,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)
        
        #RoIHeads
        box_roi_pool = RoIAlign(output_size=(7, 7), sampling_ratio=2)
        resolution = box_roi_pool.output_size[0]
        in_channels = out_channels * resolution ** 2
        mid_channels = 1024
        box_predictor = FastRCNNPredictor(in_channels, mid_channels, num_classes)

        self.head = RoIHeads(
            box_roi_pool, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_num_samples, box_positive_fraction,
            box_reg_weights, box_score_thresh, box_nms_thresh,
            box_num_detections
        )

        self.head.mask_roi_pool = RoIAlign(output_size=(14, 14), sampling_ratio=2)
        layers = (256, 256, 256, 256)
        dim_reduced = 256
        self.head.mask_predictor = MaskRCNNPredictor(out_channels, layers, dim_reduced, num_classes)

        #Transformer
        self.transformer = Transformer(
            min_size=800, max_size=1333,
            img_mean=[0.485, 0.456, 0.406],
            img_std=[0.229, 0.224, 0.225]
        )

    def forward(self, images, target=None):
        ori_image_shape = [img.shape[-2:] for img in images]

        images, target = self.transformer(images, target)
        # print("\n transformer ouput debug")
        # print("image_shape after transform:", [img.shape for img in images])
        # print("targets[0]['boxes'] after transform:", target[0]['boxes'])
        # print("targets[0]['labels']:", target[0]['labels'])

        image_shape = [img.shape[-2:] for img in images]
        batched_images = torch.stack(images, dim=0)
        feature = self.backbone(batched_images)
        
        proposal, rpn_losses = self.rpn(feature, image_shape, target)
        # print("\n rpn output debug")
        # print("Num proposals:", len(proposal[0]))
        # print("Sample proposals:", proposal[0][:5])

        # if not self.training:
        #     print(f"RPN generated {len(proposal)} proposals")

        result, roi_losses = self.head(feature, proposal, image_shape, target)

        # if not self.training:
        #     print(f"RoI Head detected {len(result['boxes'])} objects")
        #     if len(result['boxes']) > 0:
        #         print(f"Max Score: {result['scores'].max().item()}")
        
        if self.training:
            return dict(**rpn_losses, **roi_losses)
        else:
            result = self.transformer.postprocess(result, image_shape[0], ori_image_shape[0])
            return [result]
        """input: image [C, H, W]
        output:  training: cac loss (int) trong rpn
                    test:   image tensor[C, H, W]
                            target = tensor[x_min, y_min, x_max, y_max]"""

class FastRCNNPredictor(nn.Module):
    def __init__(self, in_channels, mid_channels, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, mid_channels)
        self.fc2 = nn.Linear(mid_channels, mid_channels)
        self.cls_score = nn.Linear(mid_channels, num_classes)
        self.bbox_pred = nn.Linear(mid_channels, num_classes*4)

    def forward(self, x): #x [N, C, H, W] (n: so luong propossal duoc dua vao de phan loai, c, so kenh, hw kich thuoc spatial)
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        score = self.cls_score(x)
        bbox_delta = self.bbox_pred(x)

        return score, bbox_delta
    """output:  score[N, num_classes]
                bbox_delta[N, num_classes*4]"""

class MaskRCNNPredictor(nn.Sequential):
    def __init__(self, in_channels, layers, dim_reduced, num_classes):
        d = OrderedDict()
        next_feature = in_channels
        for layer_idx, layer_features in enumerate(layers, 1):
            d['mask_fcn{}'.format(layer_idx)] = nn.Conv2d(next_feature, layer_features, 3, 1, 1)
            d['relu{}'.format(layer_idx)] = nn.ReLU(inplace=True)
            next_feature = layer_features   
        
        d['mask_conv5'] = nn.ConvTranspose2d(next_feature, dim_reduced, 2, 2, 0)
        d['relu5'] = nn.ReLU(inplace=True)
        d['mask_fcn_logits'] = nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)
        super().__init__(d)

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
    """?"""

class ResBackbone(nn.Module):
    def __init__(self, backbone_name, pretrained):
        super().__init__()
        body = models.resnet.__dict__[backbone_name](
            pretrained=pretrained, norm_layer=misc.FrozenBatchNorm2d)

        for name, parameter in body.named_parameters():
            if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False) #freeze cac tham so khong thuoc layer2,3,4, chi hoc 2,3 ,4 
               
        self.body = nn.ModuleDict(d for i, d in enumerate(body.named_children()) if i < 7)
        in_channels =1024
        self.out_channels = 256

        self.inner_block_module = nn.Conv2d(in_channels, self.out_channels, 1) # 1x1 conv giam 2048 -> 256
        self.layer_block_module = nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1) #3x3 conv giu nguyen 256, padding =1
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x): #input x tensor[N, 3, H, W]
        for module in self.body.values():
            x = module(x)
        x = self.inner_block_module(x)
        x = self.layer_block_module(x)
        return x
    """output: tensor[N, out_channels, h_out, w_out] (thuong co dinh voi resnet pretrained: h_out = h/32, w_out = w/32)"""

def maskrcnn_resnet50(pretrained, num_classes, pretrained_backbone=True):
    if pretrained:
        pretrained_backbone = True
    backbone = ResBackbone('resnet50', pretrained_backbone)
    model = MaskRCNN(backbone, num_classes)

    if pretrained:
        model_urls = {
            'maskrcnn_resnet50':
                'https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth',
        }
        #model_state_dict = load_url(model_urls['maskrcnn_resnet50'])

        # pretrained_msd = list(model_state_dict.values())
        # del_list = [i for i in range(256, 271)] + [i for i in range(273, 279)]
        # for i, del_idx in enumerate(del_list):
        #     pretrained_msd.pop(del_idx - i)

        # msd = model.state_dict()
        # skip_list = [271, 272, 273, 274, 279, 280, 281, 282, 293, 294]
        # if num_classes == 91:
        #     skip_list = [271, 272, 273, 274]
        # for i, name in enumerate(msd):
        #     if i in skip_list:
        #         continue
        #     msd[name].copy_(pretrained_msd[i])

        # model.load_state_dict(msd)

        coco_weights = load_url(model_urls['maskrcnn_resnet50'])
        model_dict = model.state_dict()

        filtered = {
            k: v for k, v in coco_weights.items()
            if k in model_dict and model_dict[k].shape == v.shape
        }

        model_dict.update(filtered)
        model.load_state_dict(model_dict)

    return model
    """luc can su dung thi load truc tiep model va truyen cac tham so can thiet"""