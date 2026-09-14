import torch
import torch.nn.functional as F
from torch import nn

from .box_ops import BoxCoder, box_iou, process_box, nms
from .utils import Matcher, BalancedPositiveNegativeSampler

class RPNHead(nn.Module):
    def __init__(self, in_channels, num_anchors):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size = 3, stride = 1, padding = 1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size = 1)
        self.bbox_pred = nn.Conv2d(in_channels, 4 * num_anchors, kernel_size = 1)

        # Khởi tạo weights bằng giá trị ngẫu nhiên từ phân phối chuẩn N(0, 0,01^2) và bias = 0
        for l in self.children():
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)
    """input class RPNHead: in_channels: int (output cua backbone+FPN)
                            num_anchors: int (so anchor per pixel của feature map)"""
    def forward(self, x): # x: [N, C, H, W] N: batch size, C: in_channels, H, W kích thước Feature map
        x = F.relu(self.conv(x))
        logits = self.cls_logits(x)
        bbox_reg = self.bbox_pred(x)
        '''
        logits	scores vật thể (objectness) cho từng anchor	[B, num_anchors, H, W]
        bbox_reg độ dịch chuyển (dx, dy, dw, dh) cho từng anchor	[B, 4 * num_anchors, H, W]'''
        return logits, bbox_reg
    """output:  logits tensor[N, num_anchors, H, W]
                bbox_reg tensor[N, num_anchors*4, H, W]"""
    
class RegionProposalNetwork(nn.Module):
    def __init__(self, anchor_generator, head,
                 fg_iou_thresh, bg_iou_thresh,
                 num_samples, positive_fraction,
                 reg_weights,
                 pre_nms_top_n, post_nms_top_n, nms_thresh):
        super().__init__()
        # head: đầu ra của RPN, bao gồm objectness và bbox regression

        self.anchor_generator = anchor_generator
        self.head = head

        self.proposal_matcher = Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=True)
        self.fg_bg_sampler = BalancedPositiveNegativeSampler(num_samples, positive_fraction)
        self.box_coder = BoxCoder(reg_weights)

        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = 1
    """input class RegionProposalNetwork: 
            + anchor_generator: object class AnchorGenerator
            + head: objeect class RPNHead
            + fg_iou_thresh: float
            + bg_iou_thresh: float
            + num_samples: int
            + positve_fraction: float
            + reg_weights: tuple(wx, wy, ww, wh)
            + pre_nms_top_n: int
            + post_nms_top_n: int
            + nms_thresh: float"""
    
    def create_proposal(self, anchor, objectness, pred_bbox_delta, image_shape):
        ''' | Tham số           | Kiểu          | Ý nghĩa                                              |
            | ----------------- | ------------- | ---------------------------------------------------- |
            | `anchor`          | Tensor [N, 4] | Tập hợp các anchor box gốc (tọa độ [x1, y1, x2, y2]) |
            | `objectness`      | Tensor [N]    | Xác suất (hoặc logit) của mỗi anchor chứa vật thể    |
            | `pred_bbox_delta` | Tensor [N, 4] | Dự đoán dịch chuyển (dx, dy, dw, dh) cho từng anchor |
            | `image_shape`     | tuple (H, W)  | Kích thước ảnh gốc để cắt box hợp lệ                 |
        '''
        # Chọn số lượng box trước và sau NMS tùy chế độ training/testing
        if self.training: 
            pre_nms_top_n = self._pre_nms_top_n['training'] #lấy ít hơn để tăng tốc độ huấn luyện
            post_nms_top_n = self._post_nms_top_n['training']
        else: 
            pre_nms_top_n = self._pre_nms_top_n['testing'] #lấy nhiều hơn để tăng độ chính xác khi đánh giá
            post_nms_top_n = self._post_nms_top_n['testing']

        pre_nms_top_n = min(objectness.shape[0], pre_nms_top_n)
        top_n_idx  = objectness.topk(pre_nms_top_n)[1]
        score = objectness[top_n_idx]
        proposal = self.box_coder.decode(pred_bbox_delta[top_n_idx], anchor[top_n_idx])

        proposal, score = process_box(proposal, score, image_shape, self.min_size)
        keep = nms(proposal, score, self.nms_thresh)[:post_nms_top_n]
        proposal = proposal[keep]
        
        return proposal # tensor([[ 34.2,  45.1, 180.3, 220.8], [210.5,  56.0, 315.9, 190.4], [ 15.0,  80.3,  95.5, 160.2], ...])

    """output: tensor[M, 4] [x_min, y_min, x_max, y_max] (proposal cuoi cung duoc chon)"""

    def compute_loss(self, objectness, pred_bbox_delta, gt_box, anchor):
        iou = box_iou(gt_box, anchor)
        label, matched_idx = self.proposal_matcher(iou)

        pos_idx, neg_idx = self.fg_bg_sampler(label)
        idx = torch.cat((pos_idx, neg_idx))
        regression_target = self.box_coder.encode(gt_box[matched_idx[pos_idx]], anchor[pos_idx])

        objectness_loss = F.binary_cross_entropy_with_logits(objectness[idx], label[idx])
        box_loss = F.l1_loss(pred_bbox_delta[pos_idx], regression_target, reduction='sum') / idx.numel()

        return objectness_loss, box_loss
    """output:  objectness_loss: float (scalar)
                box_loss: float (scalar)"""
    
    def forward(self, feature, image_shape, target=None):
        if self.training and target is not None:
            gt_box = target[0]['boxes']

        '''tạo các anchor boxes (khung tham chiếu) cho toàn bộ feature map.
            anchor: [N, 4]  (tọa độ [x1, y1, x2, y2] trên ảnh gốc)'''
        anchor = self.anchor_generator(feature, image_shape)

        '''objectness: [B, num_anchors, H, W]
           pred_bbox_delta: [B, 4*num_anchors, H, W]'''
        objectness, pred_bbox_delta = self.head(feature)
        '''[B, num_anchors, H, W] → [B, H, W, num_anchors]→ [B * H * W * num_anchors]'''
        objectness = objectness.permute(0, 2, 3, 1).flatten()
        '''→ [B * H * W * num_anchors, 4] Mỗi hàng là (dx, dy, dw, dh) của một anchor.'''
        pred_bbox_delta = pred_bbox_delta.permute(0, 2, 3, 1).reshape(-1, 4)

        proposal = self.create_proposal(anchor, objectness.detach(), pred_bbox_delta.detach(), image_shape[0])
        if self.training:
            objectness_loss, box_loss = self.compute_loss(objectness, pred_bbox_delta, gt_box, anchor)
            return proposal, dict(rpn_objectness_loss=objectness_loss, rpn_box_loss=box_loss)
        
        return proposal, {}
    """output:  proposal: tensor[M, 4]
                dict: empty khi test, tra ve cac loss (float) khi train"""