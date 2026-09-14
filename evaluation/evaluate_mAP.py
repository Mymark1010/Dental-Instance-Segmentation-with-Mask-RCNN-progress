import torch
import sys
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torchvision.ops import box_iou
from pathlib import Path

FILE = Path(__file__).resolve()
PROJECT_ROOT = FILE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from Mask_RCNN.model.mask_rcnn import maskrcnn_resnet50
from Mask_RCNN.dataset import teeth_dataset
from Mask_RCNN.dataset.torch_teeth_dataset import TorchTeethDataset
from ETE_train import collate_fn
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


# --- 1. CÁC HÀM TÍNH TOÁN mAP ---

def calculate_ap_per_class(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5):
    """
    Tính Average Precision (AP) và trả về cả Precision/Recall arrays.
    """
    if len(gt_boxes) == 0:
        return 0.0, np.array([0., 1.]), np.array([0., 0.]), 0.0 
    
    if len(pred_boxes) == 0:
        return 0.0, np.array([0., 1.]), np.array([0., 0.]), 0.0

    # 1. Sắp xếp dự đoán theo điểm tin cậy giảm dần
    sorted_indices = torch.argsort(pred_scores, descending=True)
    pred_boxes = pred_boxes[sorted_indices]
    
    # 2. Tính IoU
    ious = box_iou(pred_boxes, gt_boxes)
    
    tp = torch.zeros(len(pred_boxes))
    fp = torch.zeros(len(pred_boxes))
    gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool)

    # 3. Xác định TP/FP
    for i in range(len(pred_boxes)):
        iou_max, gt_idx = torch.max(ious[i], dim=0)
        if iou_max >= iou_threshold and not gt_matched[gt_idx]:
            tp[i] = 1
            gt_matched[gt_idx] = True
        else:
            fp[i] = 1

    # 4. Tính Precision và Recall tích lũy
    tp_cumsum = torch.cumsum(tp, dim=0)
    fp_cumsum = torch.cumsum(fp, dim=0)
    
    recalls = tp_cumsum / len(gt_boxes)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    accuracy = recalls[-1].item() if len(recalls) else 0
    
    # 5. Làm mượt
    precisions = torch.cat((torch.tensor([1.0]), precisions, torch.tensor([0.0])))
    recalls = torch.cat((torch.tensor([0.0]), recalls, torch.tensor([1.0])))
    
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = torch.max(precisions[i], precisions[i + 1])
        
    indices = torch.where(recalls[1:] != recalls[:-1])[0]
    ap = torch.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    
    return ap.item(), precisions.numpy(), recalls.numpy(), accuracy

def evaluate_mAP(model, data_loader, device, num_classes, iou_threshold=0.5):
    model.eval()
    
    # Lưu trữ dữ liệu
    class_data = {i: {'pred_boxes': [], 'pred_scores': [], 'gt_boxes': []} for i in range(1, num_classes + 1)}
    gt_class_counts = {i: 0 for i in range(1, num_classes + 1)}

    with torch.no_grad():
        for batch in tqdm(data_loader): 
            if batch is None: 
                continue 
            
            images, targets = batch
            images = [img.to(device) for img in images]
            outputs = model(images)
            
            for i, output in enumerate(outputs):
                p_boxes = output["boxes"].cpu()
                p_scores = output["scores"].cpu()
                p_labels = output["labels"].cpu()
                g_boxes = targets[i]["boxes"].cpu()
                g_labels = targets[i]["labels"].cpu()

                unique_gt_labels = torch.unique(g_labels)
                for lbl in unique_gt_labels:
                    lbl_item = lbl.item()
                    if lbl_item in gt_class_counts:
                        gt_class_counts[lbl_item] += 1
                
                for cls_id in range(1, num_classes + 1):
                    cls_mask_p = (p_labels == cls_id)
                    class_data[cls_id]['pred_boxes'].append(p_boxes[cls_mask_p])
                    class_data[cls_id]['pred_scores'].append(p_scores[cls_mask_p])
                    cls_mask_g = (g_labels == cls_id)
                    class_data[cls_id]['gt_boxes'].append(g_boxes[cls_mask_g])

    # Tính AP và lưu Precision/Recall cho từng class
    aps = []
    valid_classes = [] # Danh sách các class hợp lệ
    pr_data = {} # Dictionary lưu p, r cho từng class
    accuracies = [] # Danh sách lưu recall cao nhất trong từng class
    
    for cls_id in range(1, num_classes + 1):
        p_boxes = torch.cat(class_data[cls_id]['pred_boxes']) if class_data[cls_id]['pred_boxes'] else torch.tensor([])
        g_boxes = torch.cat(class_data[cls_id]['gt_boxes']) if class_data[cls_id]['gt_boxes'] else torch.tensor([])

        if len(g_boxes) == 0 and len(p_boxes) == 0:
            continue

        p_scores = torch.cat(class_data[cls_id]['pred_scores']) if class_data[cls_id]['pred_scores'] else torch.tensor([])
        
        ap, prec, rec, accuracy = calculate_ap_per_class(p_boxes, p_scores, g_boxes, iou_threshold)
        
        aps.append(ap)
        valid_classes.append(cls_id)
        pr_data[cls_id] = {"precision": prec, "recall": rec, "ap": ap}
        accuracies.append(accuracy)
        
    if aps:
        mAP = np.mean(aps)
    else:
        mAP = 0.0

    return mAP, aps, valid_classes, pr_data, accuracies

# --- 2. HÀM VẼ BIỂU ĐỒ AP & PR CURVE & BIỂU ĐỒ RECALL ---

def plot_mAP_results(aps_input, pr_data, accuracies, valid_class_names, full_class_names, save_dir="evaluation/evaluation_results"):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    mAP = np.mean(aps_input)
    
    # --- BIỂU ĐỒ 1: AP BAR CHART ---    
    plt.figure(figsize=(15, 6))
    ap_bars = plt.bar(valid_class_names, aps_input, color='skyblue', edgecolor='navy')
    
    # Vẽ đường kẻ đỏ dựa trên mAP thực tế (ví dụ: 0.4632)
    plt.axhline(y=mAP, color='r', linestyle='--', label=f'Overall mAP: {mAP:.4f}')
    
    plt.title('Average Precision (AP) per Class')
    plt.xlabel('Tooth Class')
    plt.ylabel('AP Score')
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xticks(rotation=90, fontsize=8)
    
    for bar in ap_bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 2), 
                 ha='center', va='bottom', fontsize=7, rotation=90)
                 
    plt.tight_layout()
    plt.savefig(save_dir / "mAP_barchart.png", dpi=300)
    print(f"Đã lưu biểu đồ mAP tại: {save_dir / 'mAP_barchart.png'}")
    plt.show()

    # --- BIỂU ĐỒ 2: PRECISION-RECALL CURVE ---
    plt.figure(figsize=(12, 8))
    
    # Tạo màu sắc đa dạng cho các đường cong
    colors = plt.cm.jet(np.linspace(0, 1, len(pr_data)))
    
    for idx, (cls_id, data) in enumerate(pr_data.items()):
        prec = data["precision"]
        rec = data["recall"]
        ap = data["ap"]
        
        # Lấy tên răng từ full_class_names (cls_id bắt đầu từ 1)
        if (cls_id - 1) < len(full_class_names):
            name = full_class_names[cls_id - 1]
        else:
            name = f"ID_{cls_id}"
            
        plt.plot(rec, prec, lw=1.5, label=f'{name} (AP={ap:.2f})', color=colors[idx], alpha=0.8)
    
    plt.title('Precision-Recall Curve per Class')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", ncol=1, fontsize='small')
    
    plt.tight_layout()
    plt.savefig(save_dir / "pr_curve.png", dpi=300)
    print(f"Đã lưu biểu đồ Precision-Recall tại: {save_dir / 'pr_curve.png'}")
    plt.show()

    # --- BIỂU ĐỒ 3: ACCURACY BAR CHART --- 
    Acc = np.mean(accuracies)
    plt.figure(figsize=(15, 6))
    accuracy_bars = plt.bar(valid_class_names, accuracies, color='skyblue', edgecolor='navy')
    plt.axhline(y=mAP, color='r', linestyle='--', label=f'Overall Acc: {Acc:.4f}')

    plt.title('Accuracy per Class')
    plt.xlabel('Tooth Class')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xticks(rotation=90, fontsize=8)

    for bar in accuracy_bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 2), 
                 ha='center', va='bottom', fontsize=7, rotation=90)
                 
    plt.tight_layout()
    plt.savefig(save_dir / "accuracy_barchart.png", dpi=300)
    print(f"Đã lưu biểu đồ mAP tại: {save_dir / 'accuracy_barchart.png'}")
    plt.show()


# --- 3. MAIN ---

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ĐƯỜNG DẪN dỮ LIỆU
    ROOT_DIR = PROJECT_ROOT / "data"
    DIR = ROOT_DIR / "general_Radiographs"
    ANN = ROOT_DIR / "general_Segmentation/teeth_polygon.json"
    WEIGHTS_PATH = "data/weights_ETE_train/baseline_epoch56.pth" 

    # Load Data
    md = teeth_dataset.TeethDataset()
    md.load_teeth(DIR, "test", ANN) 
    md.prepare()
    
    dataset = TorchTeethDataset(md, max_size=1333, augmentation=False)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # Load Model
    num_classes = md.num_classes + 1
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
    mAP, aps, valid_classes, pr_data, accuracies = evaluate_mAP(model, data_loader, device, num_classes=num_classes-1, iou_threshold=0.5)
    valid_class_names = [md.class_names[i-1] for i in valid_classes]

    # Vẽ đồ thị
    plot_mAP_results(aps, pr_data, accuracies, valid_class_names, md.class_names)

    # In kết quả dạng Text
    print(f"\n=== KẾT QUẢ ===")
    print(f"mAP: {mAP:.4f}")
    print(f"Acc: {np.mean(accuracies):.4f}")

if __name__ == "__main__":
    main()