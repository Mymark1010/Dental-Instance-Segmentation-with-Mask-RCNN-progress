import os
import json
import numpy as np
import skimage.io
import skimage.color
import skimage.draw


class TeethDataset:
    def __init__(self):
        self.image_info = []
        self.class_info = []
        self._image_ids = []

    def add_class(self, source, class_id, class_name):
        # Avoid duplicates
        for info in self.class_info:
            if info['source'] == source and info['id'] == class_id:
                return
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name
        })

    def add_image(self, source, image_id, path, **kwargs):
        info = {
            "id": image_id,
            "source": source,
            "path": path
        }
        info.update(kwargs)
        self.image_info.append(info)

    def load_teeth(self, dataset_dir, subset, annotation_json):
        assert subset in ["train", "test", "val"]
        subset_dir = os.path.join(dataset_dir, subset)

        # Linux Case-Sensitivity - Map actual filenames on disk
        # Creates map: {'45.jpg': '45.JPG'}
        actual_files = {f.lower(): f for f in os.listdir(subset_dir)}

        with open(annotation_json) as f:
            annotations = json.load(f)

        # Register all unique classes
        class_titles = set()
        for item in annotations:
            for obj in item["Label"]["objects"]:
                class_titles.add(obj["title"])
                
        # Split numeric vs alphabetic
        numeric_titles = sorted([t for t in class_titles if t.isdigit()], key=int)
        alpha_titles = sorted([t for t in class_titles if not t.isdigit()])
        
        # Create mapping
        mapping = {}

        # Numeric labels preserve original numbers
        for t in numeric_titles:
            mapping[t] = int(t)
        
        start_id = max(mapping.values(), default=0) + 1
        for idx, t in enumerate(alpha_titles):
            mapping[t] = start_id + idx

        for title, class_id in mapping.items():
            self.add_class("teeth", class_id, f"tooth_{title}")

        # Add image entries
        for item in annotations:
            filename = item["External ID"]
            filename_lower = filename.lower()

            # Linux Case-Sensitivity Check
            if filename_lower not in actual_files:
                continue

            real_filename = actual_files[filename_lower]

            image_path = os.path.join(subset_dir, real_filename)
            objects = []
            for obj in item["Label"]["objects"]:
                title = str(obj["title"]).strip()

                objects.append({
                    "class_id": mapping[title],
                    "bbox": obj["bounding box"],
                    "polygons": obj["polygons"]
                })

            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                source="teeth",
                image_id=filename,
                path=image_path,
                width=width,
                height=height,
                objects=objects
            )

    def prepare(self):
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [c["name"] for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)

        self.class_from_source_map = {
            f"{info['source']}.{info['id']}": i
            for i, info in enumerate(self.class_info)
        }

        self.image_from_source_map = {
            f"{info['source']}.{info['id']}": i
            for i, info in enumerate(self.image_info)
        }

        self.sources = list(set(info["source"] for info in self.class_info))
        self.source_class_ids = {}
        for source in self.sources:
            self.source_class_ids[source] = [
                i for i, info in enumerate(self.class_info)
                if i == 0 or info["source"] == source
            ]

    @property
    def image_ids(self):
        return self._image_ids

    def load_image(self, file_path, annotation_json):
        filename = os.path.basename(file_path).lower()
        
        with open(annotation_json) as f:
            annotations = json.load(f)

        # 1. Build the mapping for ALL classes (Required so class_ids match training)
        class_titles = set()
        for item in annotations:
            for obj in item["Label"]["objects"]:
                class_titles.add(str(obj["title"]).strip())
        
        numeric_titles = sorted([t for t in class_titles if t.isdigit()], key=int)
        alpha_titles = sorted([t for t in class_titles if not t.isdigit()])
        mapping = {t: int(t) for t in numeric_titles}
        start_id = max(mapping.values(), default=0) + 1
        for idx, t in enumerate(alpha_titles):
            mapping[t] = start_id + idx

        # 2. Find the specific item
        target_item = None
        for item in annotations:
            if item["External ID"].lower() == filename:
                target_item = item
                break
        
        if target_item is None:
            raise FileNotFoundError(f"Could not find {filename} in {annotation_json}")

        # 3. Setup image_info for just this one image
        image = skimage.io.imread(file_path)
        height, width = image.shape[:2]
        
        objects = []
        for obj in target_item["Label"]["objects"]:
            objects.append({
                "class_id": mapping[str(obj["title"]).strip()],
                "bbox": obj["bounding box"],
                "polygons": obj["polygons"]
            })

        self.add_image(
            source="teeth",
            image_id=target_item["External ID"],
            path=file_path,
            width=width,
            height=height,
            objects=objects
        )

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        if info["source"] != "teeth":
            return np.empty([0, 0, 0]), np.empty([0], np.int32)

        objects = info["objects"]
        height, width = info["height"], info["width"]
        masks = []
        class_ids = []

        for obj in objects:
            for poly in obj["polygons"]:
                if len(poly) < 3:
                    continue
                poly = np.array(poly)
                rr, cc = skimage.draw.polygon(poly[:, 1], poly[:, 0], shape=(height, width))
                mask = np.zeros((height, width), dtype=np.uint8)
                mask[rr, cc] = 1
                masks.append(mask)
                class_ids.append(obj["class_id"])

        if masks:
            mask = np.stack(masks, axis=-1)
            return mask.astype(bool), np.array(class_ids, dtype=np.int32)
        else:
            return np.empty([0, 0, 0]), np.array([], dtype=np.int32)

    def image_reference(self, image_id):
        return self.image_info[image_id]["path"]
