import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from Mask_RCNN.dataset import TeethDataset, TorchTeethDataset
import torch, matplotlib, os
from skimage.segmentation import find_boundaries
import skimage
from Mask_RCNN.model.mask_rcnn import maskrcnn_resnet50
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

class TeethVisualizer:
    """
    A utility class to visualize Ground Truth and Model Predictions 
    by integrating the dataset and the model.
    """
    def __init__(self, dataset: TorchTeethDataset, model=None):
        self.dataset = dataset
        self.model = model
        if self.model:
            self.model.eval()
            # Get num_classes from the model's classification head
            if hasattr(self.model, 'head'):
                # Custom Model (ETE)
                self.num_classes = self.model.head.box_predictor.cls_score.out_features
            elif hasattr(self.model, 'roi_heads'):
                # Torchvision Model (Baseline)
                self.num_classes = self.model.roi_heads.box_predictor.cls_score.out_features
            else:
                raise AttributeError("Không nhận diện được cấu trúc model (không có 'head' hoặc 'roi_heads')")
        else:
            self.num_classes = len(self.dataset.mds.class_info)
        self.class_map = self.dataset.mds.class_info
        self.colors = matplotlib.colormaps['hsv']

    def get_color(self, label_id: int):
        # The golden ratio conjugate
        phi = (1 + 5**0.5) / 2
        n = label_id * phi
        # Use the fractional part to get a unique hue (0.0 to 1.0)
        hue = n % 1
        # Return the color from the hsv spectrum based on this spread hue
        return self.colors(hue)

    def _get_processed_data(self, idx: int):
        """Fetches and processes data from dataset and model for a given index."""

        # 1. Get original image and metadata
        info = self.dataset.mds.image_info[idx]
        image_orig = skimage.io.imread(info["path"])
        if image_orig.ndim != 3:
            image_orig = skimage.color.gray2rgb(image_orig)
        h_orig, w_orig = image_orig.shape[:2]
        
        raw_data = self.dataset[idx]
        
        if isinstance(raw_data, list):
            image_tensor, target = raw_data[0]
        else:
            # Trường hợp dataset cũ hoặc trả về trực tiếp tuple
            image_tensor, target = raw_data

        _, h_scaled, w_scaled = image_tensor.shape

        # Calculate scales
        scale_x = w_orig / w_scaled
        scale_y = h_orig / h_scaled
        
        # Determine the model's current device (e.g., 'cuda:0' or 'cpu')
        # This is a robust way to find the device.
        if (self.model):
            model_device = next(self.model.parameters()).device
        
            # --- FIX: Move image tensor to the model's device ---
            image_tensor = image_tensor.to(model_device)

            with torch.no_grad():
                outputs = self.model([image_tensor])
            pred = outputs[0]

            num_preds = pred['boxes'].shape[0]
            if num_preds == 0: # Not detect any object
                print("No objects detected by the model.")
                # Initialize all prediction arrays to empty but correct shapes (on CPU)
                return image_orig, {"pred_masks": np.zeros((0, h_orig, w_orig)), "pred_boxes": np.zeros((0, 4)), 
                                    "pred_labels": np.array([]), "pred_scores": np.array([])}
            
            # Rescale Bboxes to Original Resolution
            pred_boxes = pred['boxes'].cpu().numpy()
            pred_boxes[:, [0, 2]] *= scale_x
            pred_boxes[:, [1, 3]] *= scale_y

            # Rescale Masks to Original Resolution
            # interpolation=False/Nearest to maintain binary mask
            pred_masks_raw = pred['masks'].cpu().squeeze(1).numpy()
            pred_masks = []
            for m in pred_masks_raw:
                # Use torch.nn.functional.interpolate for resizing without cv2
                m_tensor = torch.from_numpy(m).unsqueeze(0).unsqueeze(0)
                m_res = torch.nn.functional.interpolate(m_tensor, size=(h_orig, w_orig), mode='bilinear')
                pred_masks.append((m_res.squeeze().numpy() > 0.5).astype(np.uint8))
            pred_masks = np.stack(pred_masks)
            
            pred_labels = pred['labels'].cpu().numpy()
            pred_scores = pred['scores'].cpu().numpy()

            return image_orig, {
                "pred_masks": pred_masks,
                "pred_boxes": pred_boxes,
                "pred_labels": pred_labels,
                "pred_scores": pred_scores,
                "gt_masks": target['masks'].cpu().numpy(),
                "gt_boxes": target['boxes'].cpu().numpy(),
                "gt_labels": target['labels'].cpu().numpy()
            }

        return image_orig, {"gt_masks": target['masks'].numpy(), "gt_boxes": target['boxes'].numpy(), "gt_labels": target['labels'].numpy()}

    def visualize_masks_and_boxes(
        self, 
        idx: int = 0, 
        source: str = 'gt', 
        tooth_index: int = None, 
        score_threshold: float = 0,
        save_path: str = "result.png",
        prettier: bool = False
    ):
        """
        - source = 'gt'  ground truth của dataset
        - source = 'pred'  dự đoán của model
        - tooth_index: index của răng muốn hiển thị (bắt đầu từ 0). None để hiển thị tất cả răng
        - score_threshold: ngưỡng điểm số để lọc dự đoán (chỉ áp dụng khi source='pred')
        - save_path: nếu được cung cấp, lưu hình ảnh vào đường dẫn này thay vì hiển thị
        """
        
        image_np, data = self._get_processed_data(idx)
        
        if source == 'gt':
            masks = data['gt_masks']
            boxes = data['gt_boxes']
            labels = data['gt_labels']
            scores = None
            title_suffix = "Ground Truth"
        elif source == 'pred':
            # Filter predictions by score threshold
            valid_preds = data['pred_scores'] >= score_threshold
            masks = data['pred_masks'][valid_preds]
            boxes = data['pred_boxes'][valid_preds]
            labels = data['pred_labels'][valid_preds]
            scores = data['pred_scores'][valid_preds]
            title_suffix = f"Prediction (Score > {score_threshold:.2f})"
        else:
            raise ValueError("Source must be 'gt' or 'pred'.")

        # --- Select specific tooth if requested ---
        if tooth_index is not None:
            if 1 <= tooth_index <= len(labels):
                # L = [10, 20, 30, 40] -> L[2:3] = [30]; L[2] = 30
                masks = masks[tooth_index-1:tooth_index]
                boxes = boxes[tooth_index-1:tooth_index]
                labels = labels[tooth_index-1:tooth_index]
                scores = scores[tooth_index-1:tooth_index] if scores is not None else None
                title_suffix = f"{title_suffix} | Tooth Index: {tooth_index}"
            else:
                print(f"Warning: Tooth index {tooth_index} out of range (0 to {len(labels)-1}). Displaying all.")
                tooth_index = None # Revert to displaying all
        
        title_padding = 80
        my_dpi = 100
        h_orig, w_orig = image_np.shape[:2]
        h_total = h_orig + title_padding
        fig_width = w_orig / my_dpi
        fig_height = h_total / my_dpi

        # Display the result
        fig = plt.figure(figsize=(fig_width, fig_height), dpi=my_dpi)

        # Position the Image Axes
        # The rect is [left, bottom, width, height] in fractions of the figure
        # Image occupies from y=0 up to (h_orig / h_total)
        image_height_fraction = h_orig / h_total
        ax = fig.add_axes([0, 0, 1, image_height_fraction])
        self._plot_item(ax, image_np, masks, boxes, labels, scores, prettier=prettier, source=source)

        title = f"Image {self.dataset.mds.image_info[0]['id']} | {title_suffix}"
        title_y = 1.0 - (title_padding / 2 / h_total)
        fig.suptitle(title, y=title_y, fontsize=16, va='center', fontweight='bold')

        if save_path:
            # Create directory if it doesn't exist
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)

            plt.savefig(save_path, dpi=my_dpi) # dpi=300 for high-res save
            plt.close(fig) # Close figure to free memory and prevent showing
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()


    def _plot_item(self, ax: plt.Axes, img_np: np.ndarray, masks: np.ndarray, 
                   boxes: np.ndarray, labels: np.ndarray, scores: np.ndarray = None,
                   alpha: float = 0.5, prettier: bool = True, source: str = 'pred'):
        """Internal method for plotting the image, masks, boxes, and labels."""
        
        ax.imshow(img_np)
        ax.axis('off')

        num_objects = boxes.shape[0]

        if num_objects > 0:
            for i in range(num_objects):
                mask = masks[i]
                box = boxes[i]
                label = labels[i].item()
                
                color = self.get_color(label)
                
                # --- A. Mask Overlay ---
                # Creates a colored layer and uses the mask array as alpha/intensity
                colored_mask = np.zeros(img_np.shape, dtype=float)
                for c in range(3):
                    colored_mask[:, :, c] = color[c]
                ax.imshow(colored_mask, alpha=mask * alpha)

                if prettier:
                    if (source == 'pred'):
                        # --- Add border to mask ---
                        # Find the pixels that form the edge of the mask
                        boundaries = find_boundaries(mask, mode='inner')
                        
                        # Create a darker version of the color for the border
                        dark_color = [c * 0.5 for c in color[:3]]
                        border_overlay = np.zeros((*mask.shape, 4)) # RGBA
                        border_overlay[boundaries] = [*dark_color, 1.0] # Fully opaque dark border
                        ax.imshow(border_overlay)

                    # --- Center Label inside mask ---
                    # Find coordinates where mask is True to calculate center
                    y_coords, x_coords = np.where(mask > 0)
                    if len(x_coords) > 0 and len(y_coords) > 0:
                        center_x = np.mean(x_coords)
                        center_y = np.mean(y_coords)
                        
                        # Add text at center without score
                        ax.text(
                            center_x, center_y, str(label), 
                            color='white', fontsize=10, weight='bold',
                            ha='center', va='center' # Center alignment
                        )
                else:
                    # --- B. Bounding Box ---
                    x_min, y_min, x_max, y_max = box
                    width = x_max - x_min
                    height = y_max - y_min
                    
                    rect = Rectangle(
                        (x_min, y_min), width, height, linewidth=2, 
                        edgecolor=color, facecolor='none', linestyle='-'
                    )
                    ax.add_patch(rect)
                    
                    # --- C. Label Text ---
                    score_text = f" ({scores[i]:.2f})" if scores is not None else ""
                    
                    ax.text(
                        x_min, y_min - 5, f'{label}{score_text}', 
                        color='white', fontsize=7,
                        bbox=dict(facecolor=color[:3], alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3')
                    )

if __name__ == "__main__":

    ROOT_DIR = os.path.abspath("./")
    DIR = os.path.join(ROOT_DIR, "data/general_Radiographs")
    ANNOTATION_DIR = os.path.join(ROOT_DIR, "data/general_Segmentation/teeth_polygon.json")
    WEIGHTS_PATH = os.path.join(ROOT_DIR, "data/weights_ETE_train/baseline_epoch56.pth")

    #----------------------------------------------------
    SAVE_PATH = os.path.join(ROOT_DIR, "data/result.png")
    # SAVE_PATH = None
    TOOTH_INDEX = None
    IMAGE_NUMBER = 991
    SOURCE = 'pred'
    PRETTIER = True
    # ---------------------------------------------------

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    md = TeethDataset()
    md.load_image(os.path.join(DIR, f"test/{IMAGE_NUMBER}.JPG"), ANNOTATION_DIR)
    md.prepare() 
    dataset = TorchTeethDataset(md, max_size=None)
    
    num_classes = md.num_classes + 1 
    checkpoint = torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    if 'head.box_predictor.cls_score.weight' in state_dict:
        num_classes = state_dict['head.box_predictor.cls_score.weight'].shape[0]
    elif 'roi_heads.box_predictor.cls_score.weight' in state_dict:
        num_classes = state_dict['roi_heads.box_predictor.cls_score.weight'].shape[0]

    #custom model    
    # model = maskrcnn_resnet50(pretrained=False, num_classes=num_classes)
    # model.load_state_dict(checkpoint)
    # model.to(device)

    #pytorch baseline model
    model = maskrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.to(device)

    visualizer = TeethVisualizer(dataset=dataset, model=model)
    visualizer.visualize_masks_and_boxes(source=SOURCE, tooth_index=TOOTH_INDEX, score_threshold=0.8, save_path=SAVE_PATH, prettier=PRETTIER)