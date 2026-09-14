import torch

class Matcher:
    def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches
    """input class Matcher: high_threshold: int
                            low_threshold: int
                            allow_low_quality_matches: bool"""
    def __call__(self, iou):
        """
        Arguments:
            iou (Tensor[M, N]): containing the pairwise quality between 
            M ground-truth boxes and N predicted boxes, có dạng ma trận MxN.

        Returns:
            label (Tensor[N]): Có kích thước N, positive (1) or negative (0) label for each predicted box,
            -1 means ignoring this box.
            matched_idx (Tensor[N]): indices of gt box matched by each predicted box.
        """

        # Lấy giá trị iou lớn nhất và index tương ứng cho mỗi predicted box
        value, matched_idx = iou.max(dim=0) # Hàm max trả về giá trị max theo từng cột và index hàng của giá trị đó
        label = torch.full((iou.shape[1],), -1, dtype=torch.float, device=iou.device) #Tạo tensor label với giá trị khởi tạo -1 (ignore)
        
        label[value >= self.high_threshold] = 1
        label[value < self.low_threshold] = 0

        if self.allow_low_quality_matches:
            highest_quality = iou.max(dim=1)[0] # Lấy giá trị iou lớn nhất cho mỗi ground truth box
            gt_pred_pairs = torch.where(iou == highest_quality[:, None])[1] # Tìm predicted box tương ứng với mỗi gt box
            label[gt_pred_pairs] = 1

        return label, matched_idx
        """output:  label Tensor[N]: positive (1) or negative (0) label for each predicted box, -1 means ignoring this box.
                    matched_idx (Tensor[N]): indices of gt box matched by each predicted box.
        """
    
class BalancedPositiveNegativeSampler:
    def __init__(self, num_samples, positive_fraction):
        self.num_samples = num_samples
        self.positive_fraction = positive_fraction # Tỉ lệ positive samples trong tổng số samples
    """input class BalancedPositiveNegativeSampler: positive_fraction: float
                                                    num_samples: int"""
    def __call__(self, labels):
        # Hàm where trả về chỉ số của các phần tử thỏa mãn điều kiện
        positive = torch.where(labels == 1)[0]
        negative = torch.where(labels == 0)[0]

        num_pos = int(self.num_samples * self.positive_fraction)
        # Hàm numel() trả về số phần tử trong tensor
        num_pos = min(positive.numel(), num_pos) # Giới hạn số lượng positive samples không vượt quá số lượng thực tế
        num_neg = self.num_samples - num_pos
        num_neg = min(negative.numel(), num_neg)

        pos_perm = torch.randperm(positive.numel(), device=positive.device)[:num_pos] # Lấy ngẫu nhiên num_pos index từ positive
        neg_perm = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

        pos_idx = positive[pos_perm] # Lấy index của positive và negative samples
        neg_idx = negative[neg_perm]

        return pos_idx, neg_idx
    """output: pos_idx, neg_idx Tensor[N]"""
    
def roi_align(feature, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio):
    return torch.ops.torchvision.roi_align(feature, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio, False)
    """
    input:  feazture tensor[N, C, H, W] N anh 
            rois tensor[K, 5] [batch_idx, x1, y1, x2, y2] K so rois
            spatial_scale: float, pooled_height, pooled_width, sampling_ratio: int
    output: return pooled_rois tensor[K, C, pooled_height, pooled_width] fix size
    """
class AnchorGenerator:
    def __init__(self, sizes, ratios):
        self.sizes = sizes
        self.ratios = ratios 
        self.cell_anchor = None
        self._cache = {}
    """input class AnchorGenerator:
            + sizes anchor list(32, 64, 128, 256, 512)
            + ratios anchor list(0.5, 1.0, 2.0)
    """
    def set_cell_anchor(self, dtype, device):
        # Kiểm tra xem anchor cơ bản đã được tạo chưa
        if self.cell_anchor is not None:
            return
        
        sizes = torch.tensor(self.sizes, dtype=dtype, device=device)
        ratios = torch.tensor(self.ratios, dtype=dtype, device=device)

        h_ratios = torch.sqrt(ratios) # Tính chiều cao dựa trên tỉ lệ và kích thước anchor mặc định
        w_ratios = 1 / h_ratios # tinh chiều rộng dựa trên tỉ lệ và chiều cao anchor 

        hs = (sizes[:, None] * h_ratios[None, :]).view(-1) 
        ws = (sizes[:, None] * w_ratios[None, :]).view(-1) 
        self.cell_anchor = torch.stack([-ws, -hs, ws, hs], dim=1) / 2 # Tạo anchor cho mỗi ô trên feature map và chia đôi để lấy tọa độ từ tâm
    """cell_anchor tensor[num_anchors_per_cell, 4] [-w/2, -h/2, w/2, h/2]"""

    def grid_anchor(self, grid_size, stride):
        #grid_size (H, W), stride (stride_x, stride_y)
        dtype, device = self.cell_anchor.dtype, self.cell_anchor.device
        shift_x = torch.arange(0, grid_size[1], dtype=dtype, device=device) * stride[1] # Tạo lưới các điểm tâm anchor trên feature map
        shift_y = torch.arange(0, grid_size[0], dtype=dtype, device=device) * stride[0]
        y, x = torch.meshgrid(shift_y, shift_x) # Tạo lưới 2D từ các điểm tâm
        x = x.reshape(-1) #-1 nghĩa là tự tính chiều này sao cho tổng số phần tử giữ nguyên.
        y = y.reshape(-1) # Chuyển lưới 2D thành 1D
        shift = torch.stack((x, y, x, y), dim=1).reshape(-1, 1, 4) # Tạo tensor shift để dịch chuyển anchor về đúng vị trí trên feature map
        anchor = (shift + self.cell_anchor).reshape(-1, 4) # Dịch chuyển anchor về đúng vị trí trên feature map
        return anchor
    """output: tensor [H*W*num_anchors_per_cell, 4] [x1, y1, x2, y2]"""

    def cached_grid_anchor(self, grid_size, stride):
        key = grid_size + stride #key: tuple[H, W, stride_h, stride_w]
        if key in self._cache:
            return self._cache[key] # Trả về anchor đã được lưu trong cache nếu có
        anchor = self.grid_anchor(grid_size, stride) # Tạo anchor cho kích thước lưới và stride đã cho
        
        if len(self._cache) >= 3:
            self._cache.clear() # Giới hạn kích thước cache để tránh sử dụng quá nhiều bộ nhớ
        self._cache[key] = anchor # Lưu anchor vào cache
        return anchor
    """output: tensor[N*A, 4] [x_min, y_min, x_max, y_max]
                N = H*W so cell tren feature map
                A = ratios * sizes so anchor tren moi cell"""
    
    def __call__(self, feature, image_size):
        dtype, device = feature.dtype, feature.device
        grid_size = tuple(feature.shape[-2:]) # Lấy kích thước của feature map
        # Lấy ảnh đầu tiên (hoặc batch) để tính stride
        h, w = image_size[0] if isinstance(image_size, (list, tuple)) else image_size
        h, w = int(h), int(w)
        image_size = (h, w)

        grid_size = tuple(int(x) for x in grid_size)
        stride = tuple(i//g for i, g in zip(image_size, grid_size)) # Tính stride dựa trên kích thước ảnh và kích thước feature map

        self.set_cell_anchor(dtype, device) # Thiết lập anchor cho mỗi ô trên feature map
        anchor = self.cached_grid_anchor(grid_size, stride) # Lấy anchor từ cache hoặc tạo mới nếu chưa có
        return anchor
    