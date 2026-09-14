import math
import torch
import torch.nn.functional as F

class Transformer:
    def __init__(self, min_size, max_size, img_mean, img_std):
        self.min_size = min_size
        self.max_size = max_size
        self.img_mean = img_mean
        self.img_std = img_std
    """input của class Transformer: 
            min_size: 1 gia tri int
            max_size: 1 gia tri int 
            img_mean: 1 tensor[3] la mean cua 3 kenh (r,g,b)
            img_std: 1 tensor[3] la std cua 3 kenh"""
    def __call__(self, images, targets):
        images = [self.normalize(img) for img in images]
        images, targets = self.resize(images, targets)
        images = [self.batched_image(img) for img in images]

        return images, targets
    """output:  image tensor[C, H, W]
                target = tensor[x_min, y_min, x_max, y_max]"""
    
    def normalize(self, image):
        if image.shape[0] == 1: #image co dang tensor [C, H, W]
            image = image.repeat(3, 1, 1) # da so backbone yeu cau anh vao co 3 kenh (repeat cac anh grayvajay)
        dtype, device = image.dtype, image.device
        mean = torch.tensor(self.img_mean, dtype=dtype, device=device)
        std = torch.tensor(self.img_std, dtype=dtype, device=device)
        return (image - mean[:, None, None]) / std[:, None, None]
    """output: image sau khi chuan hoa van la tensor [C, H_normalized, W_normalized]"""

    def resize(self, images, targets):
        new_images = []
        if targets is None:
            targets = [None] * len(images)
        new_targets = []
        for img, tg in zip(images, targets):
            ori_h, ori_w = img.shape[-2:]
            min_size = float(min(ori_h, ori_w))
            max_size = float(max(ori_h, ori_w))
            scale_factor = min(self.min_size / min_size, self.max_size / max_size)
            new_h, new_w = round(ori_h * scale_factor), round(ori_w * scale_factor)
            img_resized = F.interpolate(img[None], size=(new_h, new_w), mode="bilinear", align_corners=False)[0]
            new_images.append(img_resized)

            if tg is not None:
                box = tg['boxes']
                box[:, [0, 2]] = box[:, [0, 2]] * new_w / ori_w
                box[:, [1, 3]] = box[:, [1, 3]] * new_h / ori_h
                tg['boxes'] = box
                if 'masks' in tg:
                    mask = tg['masks']
                    if mask.numel() > 0: # Chỉ resize nếu có ít nhất 1 mask
                        with torch.no_grad():
                            mask = F.interpolate(mask[None].float(), size=(new_h, new_w), mode='nearest')[0].byte().cpu()
                    else:
                        # Nếu không có mask, tạo tensor rỗng với kích thước mới
                        mask = torch.zeros((0, new_h, new_w), dtype=torch.uint8).cpu()
                tg['masks'] = mask
            new_targets.append(tg)
        return new_images, new_targets
    """output: image tensor[C, H, W]
               target = tensor[x_min, y_min, x_max, y_max]"""

    def batched_image(self, image, stride=32):
        size = image.shape[-2:]
        max_size = tuple(math.ceil(s / stride) * stride for s in size) # anh duoc pad len kich thuoc cao hon: H_pad, W_pad chia het cho stride
        batch_shape = (image.shape[-3],) + max_size
        batched_img = image.new_full(batch_shape, 0) # pad with zeros
        batched_img[:, :image.shape[-2], :image.shape[-1]] = image
        
        return batched_img
    """output: tensor co them batch dimension (1, C, H_pad, W_pad)"""

    def postprocess(self, result, image_shape, ori_image_shape):
        box = result['boxes']
        box[:, [0, 2]] = box[:, [0, 2]] * ori_image_shape[1] / image_shape[1] # width
        box[:, [1, 3]] = box[:, [1, 3]] * ori_image_shape[0] / image_shape[0] # height
        result['boxes'] = box

        if 'masks' in result:
            mask = result['masks']
            mask = paste_masks_in_image(mask, box, 1, ori_image_shape)
            result['masks'] = mask
        return result
    """output: result['boxes']: tensor[N, 4] ([x_min, y_min, x_max, y_max])
               result['mask']: tensor[N, H_ori, W_ori]"""
def expand_detection(mask, box, padding): #pading: int so pixel them quanh mask
    M = mask.shape[-1]
    scale = (M + 2 * padding) / M # scale factor to expand the mask by padding
    padding_mask = torch.nn.functional.pad(mask, (padding,) * 4) # pad the mask

    w_half = (box[:, 2] - box[:, 0]) * 0.5
    h_half = (box[:, 3] - box[:, 1]) * 0.5
    x_c = (box[:, 2] + box[:, 0]) * 0.5 # center x
    y_c = (box[:, 3] + box[:, 1]) * 0.5

    w_half = w_half * scale
    h_half = h_half * scale
    
    box_exp = torch.zeros_like(box) # expanded boxes    
    box_exp[:, 0] = x_c - w_half # left
    box_exp[:, 2] = x_c + w_half
    box_exp[:, 1] = y_c - h_half
    box_exp[:, 3] = y_c + h_half
    return padding_mask, box_exp.to(torch.int64)
    """output:  padding_mask: tensor[N, M, M] M la kich thuoc mask sau resize (28x28)
                box_exp: tensor[N, 4] [x_min, y_min, x_max, y_max] """
    
def paste_masks_in_image(mask, box, padding, image_shape):
    mask, box = expand_detection(mask, box, padding)

    N = mask.shape[0]
    size = (N,) + tuple(image_shape) # size of the final image masks
    im_mask = torch.zeros(size, dtype=mask.dtype, device=mask.device)
    for m, b, im in zip(mask, box, im_mask):
        # Resize the mask to the size of the bounding box
        b = b.tolist()
        w = max(b[2] - b[0], 1)
        h = max(b[3] - b[1], 1)

        m = F.interpolate(m[None, None], size=(h, w), mode="bilinear", align_corners=False)[0][0]

        x1 = max(b[0], 0)
        y1 = max(b[1], 0)
        x2 = min(b[2], image_shape[1])
        y2 = min(b[3], image_shape[0])

        im[y1:y2, x1:x2] = m[(y1 - b[1]):(y2 - b[1]), (x1 - b[0]):(x2 - b[0])]
    return im_mask
    """outout: im_mask tensor[N, H, W] N object, va kich thuoc tung mask""" 
