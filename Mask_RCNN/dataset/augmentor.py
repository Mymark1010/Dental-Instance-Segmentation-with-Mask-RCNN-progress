import albumentations as A
import numpy as np

class TeethAugmentor:
    def __init__(self):        
        self.transform = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.15,contrast_limit=0.15,p=0.4),
            A.GaussNoise( std_range=(0.01, 0.03), p=0.2),
            A.GaussianBlur(blur_limit=3, p=0.2),
            A.Affine(translate_percent=0.02,scale=(0.97, 1.03),rotate=(-5, 5),p=0.3),
            A.GridDistortion(num_steps=5, distort_limit=0.05, p=0.2),
            ],
            bbox_params=A.BboxParams(format='pascal_voc',label_fields=['class_labels'],min_visibility=0.3))

    def __call__(self, image, masks, bboxes, labels):
        try:
            if len(bboxes) == 0:
                return image, masks, bboxes, labels

            augmented = self.transform(image=image, masks=masks, bboxes=bboxes, class_labels=labels)
            
            # Lọc box rác lần cuối
            aug_bboxes = np.array(augmented['bboxes'])
            if len(aug_bboxes) > 0:
                valid_inds = (aug_bboxes[:, 2] > aug_bboxes[:, 0]) & (aug_bboxes[:, 3] > aug_bboxes[:, 1])
                if not np.all(valid_inds):
                    augmented['bboxes'] = [b for i, b in enumerate(augmented['bboxes']) if valid_inds[i]]
                    augmented['class_labels'] = [l for i, l in enumerate(augmented['class_labels']) if valid_inds[i]]
                    augmented['masks'] = [m for i, m in enumerate(augmented['masks']) if valid_inds[i]]

            return augmented['image'], augmented['masks'], augmented['bboxes'], augmented['class_labels']
        
        except Exception as e:
            return image, masks, bboxes, labels