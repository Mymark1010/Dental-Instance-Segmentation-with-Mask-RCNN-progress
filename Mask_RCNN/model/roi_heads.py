import torch
import torch.nn.functional as F
from torch import nn

from .utils import Matcher, BalancedPositiveNegativeSampler, roi_align
from .box_ops import BoxCoder, box_iou, process_box, nms

def fastrcnn_loss(class_logits, box_regression, label, regression_targets):
    # print("\n roi box loss degub ===")
    # print("class_logits mean:", class_logits.mean().item())
    # print("labels unique:", label.unique())
    # print("box_regression stats:", box_regression.min().item(), box_regression.max().item())
    # print("regression_targets stats:", regression_targets.min().item(), regression_targets.max().item())

    num_classes = class_logits.shape[1] 
    device = class_logits.device
    weights = torch.ones(num_classes).to(device)
    
    rare_ids = [1, 16] + list(range(33, 53))
    weights[rare_ids] = 2.0 # Phạt nặng gấp 5 lần nếu sai răng hiếm
    classification_loss = F.cross_entropy(class_logits, label, weight=weights)

    pos_idx = torch.where(label > 0)[0]
    if len(pos_idx) == 0:
        return classification_loss, torch.tensor(0., device=label.device)
    
    box_regression = box_regression.reshape(-1, class_logits.shape[1], 4)
    matched_box_reg = box_regression[pos_idx, label[pos_idx]]
    box_loss = F.smooth_l1_loss(
        matched_box_reg,
        regression_targets[pos_idx],
        reduction='sum'
    )

    box_loss = box_loss / label.numel()

    return classification_loss, box_loss
"""output:  classification_loss: float (scalar)
            box_reg_loss: float (scalar)"""

# def tversky_loss(inputs, targets, alpha=0.3, beta=0.7, smooth=1.0):
#     """
#     alpha: Trọng số cho False Positives (FP)
#     beta: Trọng số cho False Negatives (FN) -> Tăng beta để giảm bỏ sót răng.
#     """
#     inputs = torch.sigmoid(inputs) # Chuyển Logits về xác suất [0, 1]
#     inputs = inputs.view(-1)
#     targets = targets.view(-1)
    
#     # True Positives (TP), False Positives (FP), False Negatives (FN)
#     TP = (inputs * targets).sum()    
#     FP = (inputs * (1 - targets)).sum()
#     FN = ((1 - inputs) * targets).sum()
    
#     tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)  
    
#     return 1 - tversky

def maskrcnn_loss(mask_logit, proposal, matched_idx, label, gt_mask):
    matched_idx = matched_idx[:, None].to(proposal)
    roi = torch.cat((matched_idx, proposal), dim=1)

    M = mask_logit.shape[-1]
    gt_mask = gt_mask[:, None].to(roi)
    mask_target = roi_align(gt_mask, roi, 1., M, M, -1)[:, 0]

    idx = torch.arange(label.shape[0], device=label.device)

    relevant_logits = mask_logit[idx, label]
    bce_loss = F.binary_cross_entropy_with_logits(relevant_logits, mask_target)
    # d_loss = dice_loss(relevant_logits, mask_target)
    # t_loss = tversky_loss(relevant_logits, mask_target, alpha=0.3, beta=0.7)

    return bce_loss # + t_loss
"""output: mask_loss: float (scalar)"""

class RoIHeads(nn.Module):
    def __init__(self, box_roi_pool, box_predictor,
                 fg_iou_thresh, bg_iou_thresh,
                 num_samples, positive_fraction,
                 reg_weights, score_thresh, nms_thresh, num_detections):
        super().__init__()
        self.box_roi_pool = box_roi_pool
        self.box_predictor = box_predictor

        self.mask_roi_pool = None
        self.mask_predictor = None
        self.proposal_matcher = Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=False)
        self.fg_bg_sampler = BalancedPositiveNegativeSampler(num_samples, positive_fraction)
        self.box_coder = BoxCoder(reg_weights)

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.num_detections = num_detections
        self.min_size = 1
    """input class RoIHeads:
            + box_roi_pool: duoc khoi tao san trong nn.Module
            + box_predictor: duoc khoi tao san trong nn.Module
            + fg_iou_thresh: float
            + bg_iou_thresh: float
            + num_samples: int so anchor de tinh loss
            + positive_fraction: float
            + reg_weights: tuple[float](wx, wy, ww, wh) trong so offset bbox
            + score_thresh: float
            + nms_thresh: float
            + num_detection: int"""
    
    def has_mask(self):
        if self.mask_roi_pool is None:
            return False
        if self.mask_predictor is None:
            return False
        return True
    
    def select_training_samples(self, proposal, target):
        # print("\n roi select_training_samples debug")
        # print("proposals shape:", proposal[0].shape)
        # print("GT boxes:", target[0]["boxes"])

        gt_box = target[0]['boxes']
        gt_label = target[0]['labels']
        proposal = torch.cat((proposal, gt_box))

        iou = box_iou(gt_box, proposal)
        pos_neg_label, matched_idx = self.proposal_matcher(iou)
        pos_idx, neg_idx = self.fg_bg_sampler(pos_neg_label)
        idx = torch.cat((pos_idx, neg_idx))

        regression_target = self.box_coder.encode(gt_box[matched_idx[pos_idx]], proposal[pos_idx])
        proposal = proposal[idx]
        matched_idx = matched_idx[idx]
        label = gt_label[matched_idx]
        num_pos = pos_idx.shape[0]
        label[num_pos:] = 0
        #print("labels_after_matching:", label)

        return proposal, matched_idx, label, regression_target
    """output:
            + proposal: tensor[N, 4]
            + matched_idx: tensor[N]
            + label: Tensor[N]
            + regression_target: Tensor[num_pos, 4] chi tinh positive proposal"""

    def fastrcnn_inference(self, class_logit, box_regression, proposal, image_shape):
        N, num_classes = class_logit.shape

        device = class_logit.device
        pred_score = F.softmax(class_logit, dim=-1)
        box_regression = box_regression.reshape(N, -1, 4)

        boxes = []
        labels = []
        scores = []
        for l in range(1, num_classes):
            score, box_delta = pred_score[:, l], box_regression[:, l]

            keep = score >= self.score_thresh
            box, score, box_delta = proposal[keep], score[keep], box_delta[keep]
            box = self.box_coder.decode(box_delta, box)

            box, score  = process_box(box, score, image_shape[0], self.min_size)
            keep = nms(box, score, self.nms_thresh)[:self.num_detections] # co the thay bang batched_nms o day
            box, score = box[keep], score[keep]
            label = torch.full((len(keep),), l, dtype=keep.dtype, device=device)

            boxes.append(box)
            labels.append(label)
            scores.append(score)

        results = dict(boxes=torch.cat(boxes), labels=torch.cat(labels), scores=torch.cat(scores))
        return results
    """output: 
            + result['boxes']: tensor[M, 4] M la so bbox sau nms, process, topk...
            + result['labels']: tensor[M]
            + result['scores']: tensor[M]"""

    def forward(self, feature, proposal, image_shape, target):
        if self.training:
            proposal, matched_idx, label, regression_target = self.select_training_samples(proposal, target)

        box_feature = self.box_roi_pool(feature, proposal, image_shape)
        class_logit, box_regression = self.box_predictor(box_feature)

        result, losses = {}, {}
        if self.training:
            classifier_loss, box_reg_loss = fastrcnn_loss(class_logit, box_regression, label, regression_target)
            losses = dict(roi_classifier_loss=classifier_loss, roi_box_loss=box_reg_loss)
        else:
            result = self.fastrcnn_inference(class_logit, box_regression, proposal, image_shape)

        if self.has_mask():
            if self.training:
                num_pos = regression_target.shape[0]

                mask_proposal = proposal[:num_pos]
                pos_matched_idx = matched_idx[:num_pos]
                mask_label = label[:num_pos]

                if mask_proposal.shape[0] == 0:
                    losses.update(dict(roi_mask_loss=torch.tensor(0)))
                    return result, losses
                
            else:
                mask_proposal = result['boxes']

                if mask_proposal.shape[0] == 0:
                    result.update(dict(masks=torch.empty((0, 28, 28))))
                    return result, losses
                
            mask_feature = self.mask_roi_pool(feature, mask_proposal, image_shape)
            mask_logit = self.mask_predictor(mask_feature)

            if self.training:
                gt_mask = target[0]['masks']
                mask_loss = maskrcnn_loss(mask_logit, mask_proposal, pos_matched_idx, mask_label, gt_mask)
                losses.update(dict(roi_mask_loss=mask_loss))
            else:
                label = result['labels']
                idx = torch.arange(label.shape[0], device=label.device)
                mask_logit = mask_logit[idx, label]

                mask_prob = mask_logit.sigmoid()
                result.update(dict(masks = mask_prob))

        return result, losses
    """output: 
        training:   + result: empty
                    + losses: dict voi cac roi_classifier_loss, roi_box_loss, roi_mask_loss: float (scalar)
        eval/test:  + result: dict voi cac boxes, labels, scores (output cua ham fastrcnn_inference) + mask[N, H, W] neu co
                    + losses: empty"""