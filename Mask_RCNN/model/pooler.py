import torch
import math

from .utils import roi_align

class RoIAlign:
    def __init__(self, output_size, sampling_ratio):
        self.output_size = output_size
        self.sampling_ratio = sampling_ratio
        self.spatial_scale = None

    def setup_scale(self, feature_shape, image_shape):
        if self.spatial_scale is not None:
            return
        possible_scale = []

        for f, i in zip(feature_shape, image_shape):
            stack = 2 ** int(math.log2(f/i))
            possible_scale.append(stack)
        assert possible_scale[0] == possible_scale[1] # anh khong vuong (H khac W) feature map khong dong ty le -> crash
        self.spatial_scale = possible_scale[0]
        """Nếu feature map = 56x56, ảnh gốc = 224x224
        s1/s2 = 56/224 = 0.25
        math.log2(0.25) = -2
        int(-2) = -2
        2 ** -2 = 0.25 → spatial_scale = 0.25"""

    def __call__(self, feature, proposal, image_shape):
        idx = proposal.new_full((proposal.shape[0], 1), 0)   #tensor shape [K, 1] với tất cả giá trị = 0 proposal
        roi = torch.cat((idx, proposal), dim=1) # Kết hợp id và proposal để tạo thành rois
        self.setup_scale(feature.shape[-2:], image_shape[0])
        return roi_align(feature.to(roi), roi, self.spatial_scale, self.output_size[0], self.output_size[1], self.sampling_ratio)
    
    """    
    input class RoIAlign:
    + feature: Tensor [N, C, H, W] - Feature maps
    + proposal: Tensor [K, 4] - Proposal boxes (x1, y1, x2, y2)
    + image_shape: [H, W] - Kích thước ảnh gốc

    output:
    tensor[K, C, output_height, output_width] - Cac vung dac trung da duoc align
    """