import sys
import os
from pathlib import Path

FILE = Path(__file__).resolve()
PROJECT_ROOT = FILE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from Mask_RCNN.model.mask_rcnn import maskrcnn_resnet50 
from Mask_RCNN.dataset import teeth_dataset
from Mask_RCNN.dataset.torch_teeth_dataset import TorchTeethDataset
from ETE_train import collate_fn
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# --- 1. CÁC HÀM TÍNH TOÁN ---

def compute_dice_coefficient(pred_mask, gt_mask):
    """
    Tính chỉ số Dice cho 2 mask nhị phân.
    """
    gt_mask = gt_mask.to(pred_mask.device)
    pred_mask = pred_mask > 0
    gt_mask = gt_mask > 0
    
    intersection = (pred_mask & gt_mask).sum()
    sum_area = pred_mask.sum() + gt_mask.sum()
    
    if sum_area == 0:
        return 1.0  # Cả 2 đều nền đen -> Đúng
    
    dice = (2.0 * intersection) / sum_area
    return dice.item() if isinstance(dice, torch.Tensor) else dice

def evaluate_model(model, data_loader, device, num_classes_with_bg, score_thresh=0.01):
    model.eval()
    num_teeth_classes = num_classes_with_bg - 1 
    dice_per_class = {i: [] for i in range(1, num_teeth_classes + 1)}
    
    print("Đang đánh giá model...")
    with torch.no_grad():
        for batch in tqdm(data_loader):
            if batch is None: # Kiểm tra nếu collate_fn trả về None
                continue
                
            images, targets = batch 
            images = [img.to(device) for img in images]
            outputs = model(images)
            
            for i, output in enumerate(outputs):
                target = targets[i]
                
                pred_labels = output["labels"]
                pred_scores = output["scores"]
                pred_masks = output["masks"]
                gt_labels = target["labels"]
                gt_masks = target["masks"]
                # Lấy kích thước thực tế của mask từ Ground Truth
                h, w = gt_masks.shape[-2:] 

                # Lọc theo threshold
                keep_idx = pred_scores >= score_thresh
                pred_labels = pred_labels[keep_idx]
                pred_masks = pred_masks[keep_idx]
                
                # Binarize masks
                pred_masks = (pred_masks > 0.5).squeeze(1).byte()
                gt_masks = (gt_masks > 0).byte()
                
                for cls_id in range(1, num_teeth_classes + 1):
                    # Gộp mask GT
                    gt_idx = (gt_labels == cls_id).nonzero(as_tuple=True)[0]
                    if len(gt_idx) > 0:
                        cls_gt_mask = torch.any(gt_masks[gt_idx], dim=0)
                    else:
                        cls_gt_mask = torch.zeros((h, w), device=device, dtype=torch.uint8)
                    # Gộp mask Pred
                    pred_idx = (pred_labels == cls_id).nonzero(as_tuple=True)[0]
                    if len(pred_idx) > 0:
                        cls_pred_mask = torch.any(pred_masks[pred_idx], dim=0)
                    else:
                        cls_pred_mask = torch.zeros((h, w), device=device, dtype=torch.uint8)
                    
                    dice = compute_dice_coefficient(cls_pred_mask, cls_gt_mask)
                    
                    # Chỉ lưu nếu có dữ liệu (tránh nhiễu từ các class ko bao giờ xuất hiện)
                    if cls_gt_mask.sum() > 0 or cls_pred_mask.sum() > 0:
                        dice_per_class[cls_id].append(dice)
    return dice_per_class

# --- 2. HÀM VẼ ĐỒ THỊ ---

def plot_results(dice_per_class, class_names, save_dir="evaluation/evaluation_results"):
    """
    Vẽ biểu đồ Box Plot và Bar Chart từ kết quả Dice.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Chuẩn bị dữ liệu
    labels = []
    data = []
    means = []
    
    sorted_keys = sorted(dice_per_class.keys())
    
    for cls_id in sorted_keys:
        scores = dice_per_class[cls_id]
        if len(scores) > 0:
            name = class_names[cls_id - 1] 
            labels.append(name)
            data.append(scores)
            means.append(np.mean(scores))
    
    if not data:
        print("Không có dữ liệu để vẽ đồ thị.")
        return

    # --- BIỂU ĐỒ 1: BOX PLOT (Phân bố điểm số) ---
    plt.figure(figsize=(15, 6))
    plt.boxplot(data, labels=labels, patch_artist=True, 
                boxprops=dict(facecolor='lightblue', color='blue'),
                medianprops=dict(color='red'))
    
    plt.title('Dice Score Distribution per Tooth Class (Box Plot)')
    plt.xlabel('Tooth Class ID')
    plt.ylabel('Dice Coefficient')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=90, fontsize=8)
    
    # Lưu và hiển thị
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "dice_boxplot.png"), dpi=300)
    print(f"Đã lưu biểu đồ Box Plot tại: {os.path.join(save_dir, 'dice_boxplot.png')}")
    plt.show()

    # --- BIỂU ĐỒ 2: BAR CHART (Điểm trung bình) ---
    plt.figure(figsize=(15, 6))
    bars = plt.bar(labels, means, color='skyblue', edgecolor='navy')
    
    # Thêm đường trung bình tổng thể
    overall_mean = np.mean([item for sublist in data for item in sublist])
    plt.axhline(y=overall_mean, color='r', linestyle='--', label=f'Overall Mean: {overall_mean:.2f}')
    
    plt.title('Average Dice Score per Tooth Class')
    plt.xlabel('Tooth Class ID')
    plt.ylabel('Mean Dice Score')
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xticks(rotation=90, fontsize=8)
    
    # Hiển thị số trên đầu cột
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 2), 
                 ha='center', va='bottom', fontsize=7, rotation=90)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "dice_barchart.png"), dpi=300)
    print(f"Đã lưu biểu đồ Bar Chart tại: {os.path.join(save_dir, 'dice_barchart.png')}")
    plt.show()

# --- 3. MAIN ---

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ĐƯỜNG DẪN DỮ LIỆU
    ROOT_DIR = PROJECT_ROOT / "data"
    DIR = ROOT_DIR / "general_Radiographs"
    ANN = ROOT_DIR / "general_Segmentation/teeth_polygon.json"
    WEIGHTS_PATH = "data/weights_ETE_train/baseline_epoch56.pth" 

    # Load Data
    md = teeth_dataset.TeethDataset()
    md.load_teeth(DIR, "test", ANN) 
    md.prepare()
    
    dataset = TorchTeethDataset(md, max_size=1333, augmentation=False)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=collate_fn
    )

    # Load Model
    num_classes = md.num_classes + 1

    # model tự train
    # model = maskrcnn_resnet50(pretrained=True, num_classes=num_classes)
    # model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device, weights_only=True))
    # model.to(device)

    #model baseline
    model = maskrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    print(f"Loading weights from: {WEIGHTS_PATH}")
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.to(device)

    # Đánh giá
    dice_results = evaluate_model(model, data_loader, device, num_classes_with_bg=num_classes)

    # Vẽ và lưu đồ thị
    plot_results(dice_results, md.class_names)

    # # In kết quả dạng Text
    print("\n===KẾT QUẢ ===")
    all_scores = [s for scores in dice_results.values() for s in scores]
    if all_scores:
        print(f"Overall Mean Dice: {np.mean(all_scores):.4f}")
    else:
        print("Không có dữ liệu hợp lệ để tính toán.")
    
if __name__ == "__main__":
    main()