import torch
import numpy as np
from torch.utils.data import Dataset
from Mask_RCNN.dataset.teeth_dataset import TeethDataset
import cv2
from utils.converter import polygons2mask
from Mask_RCNN.dataset.augmentor import TeethAugmentor

# File: Mask_RCNN/dataset/torch_teeth_dataset.py

# ... (các import và phần đầu class TorchTeethDataset giữ nguyên) ...

class TorchTeethDataset(Dataset):
    def __init__(self, mrcnn_dataset: TeethDataset, max_size=1333, augmentation=False, start_aug_epoch=10):
        self.mds = mrcnn_dataset
        self.max_size = max_size
        self.augmentation = augmentation
        self.augmentor = TeethAugmentor()
        
        self.current_epoch = 0
        self.start_aug_epoch = start_aug_epoch 

    def __len__(self):
        return len(self.mds.image_ids)

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def process_image(self, img_np, masks, bboxes, labels):
        h, w = img_np.shape[:2]

        if self.max_size:
            scale = self.max_size / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            image_resized = cv2.resize(img_np, (new_w, new_h))
        else:
            new_w, new_h = w, h
            image_resized = img_np
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0

        all_masks = []
        all_boxes = []
        all_labels = []

        for m, b, l in zip(masks, bboxes, labels):
            if self.max_size:
                m_resized = cv2.resize(m, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                scaled_bbox = [b[0] * scale, b[1] * scale, b[2] * scale, b[3] * scale]
            else:
                m_resized = m
                scaled_bbox = b
            all_masks.append(m_resized)
            all_boxes.append(scaled_bbox)
            all_labels.append(l)

        if not all_masks:
            return None

        masks_t = torch.tensor(np.stack(all_masks)).float()
        boxes_t = torch.tensor(all_boxes, dtype=torch.float32)
        labels_t = torch.tensor(all_labels, dtype=torch.int64)
        
        boxes_t[:, 0] = boxes_t[:, 0].clamp(min=0, max=new_w)
        boxes_t[:, 1] = boxes_t[:, 1].clamp(min=0, max=new_h)
        boxes_t[:, 2] = boxes_t[:, 2].clamp(min=0, max=new_w)
        boxes_t[:, 3] = boxes_t[:, 3].clamp(min=0, max=new_h)

        valid = (boxes_t[:, 2] > boxes_t[:, 0]) & (boxes_t[:, 3] > boxes_t[:, 1])
        
        if valid.sum() == 0:
            return None

        target = {
            "boxes": boxes_t[valid],
            "labels": labels_t[valid],
            "masks": masks_t[valid]
        }
        return image_tensor, target

    def __getitem__(self, idx):
        info = self.mds.image_info[idx]
        image_raw = cv2.imread(info["path"])
        image_raw = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = image_raw.shape[:2]

        lab = cv2.cvtColor(image_raw, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        orig_masks = []
        orig_bboxes = []
        orig_labels = []

        for obj in info["objects"]:
            m = polygons2mask(img_shape=(h_orig, w_orig), polygons=obj["polygons"], scale=1.0)
            orig_masks.append(m)
            y_min, x_min, y_max, x_max = obj["bbox"]
            orig_bboxes.append([x_min, y_min, x_max, y_max])
            orig_labels.append(obj["class_id"])

        # Tạo một list các sample để trả về
        samples = []

        # 1. Xử lý ảnh GỐC
        original_sample = self.process_image(img_np, orig_masks, orig_bboxes, orig_labels)
        if original_sample is not None:
            samples.append(original_sample)

        if (not self.max_size):
            return samples
        
        # 2. Xử lý ảnh AUGMENTATION (nếu đủ điều kiện epoch)
        if self.augmentation and self.current_epoch >= self.start_aug_epoch and len(orig_bboxes) > 0:
            aug_img, aug_masks, aug_bboxes, aug_labels = self.augmentor(
                image=img_np, masks=orig_masks, bboxes=orig_bboxes, labels=orig_labels
            )
            augmented_sample = self.process_image(aug_img, aug_masks, aug_bboxes, aug_labels)
            if augmented_sample is not None:
                samples.append(augmented_sample)
        
        # Trả về một list các sample. collate_fn sẽ nhận list này.
        # Nếu không có sample nào hợp lệ, trả về None để collate_fn lọc.
        return samples if samples else None