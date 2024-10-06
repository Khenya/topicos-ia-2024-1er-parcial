from typing import Optional, List
from ultralytics import YOLO
import numpy as np
import cv2
from shapely.geometry import Polygon, box
from src.models import Detection, PredictionType, Segmentation
from src.config import get_settings

SETTINGS = get_settings()

def match_gun_bbox(segment: List[List[int]], bboxes: List[List[int]], max_distance: int = 10) -> Optional[List[int]]:
    matched_box = None
    min_distance = float('inf')

    segment_poly = Polygon(segment)
    
    for bbox in bboxes:
        bbox_poly = box(*bbox)
        distance = segment_poly.distance(bbox_poly)
        
        if distance < min_distance and distance <= max_distance:
            min_distance = distance
            matched_box = bbox
    
    return matched_box

def annotate_detection(image_array: np.ndarray, detection: Detection) -> np.ndarray:
    ann_color = (255, 0, 0)
    annotated_img = image_array.copy()
    
    for label, conf, box in zip(detection.labels, detection.confidences, detection.boxes):
        x1, y1, x2, y2 = map(int, box)  

        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), ann_color, 3)
        cv2.putText(
            annotated_img,
            f"{label}: {conf:.1f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            ann_color,
            2,
        )
    return annotated_img

def annotate_segmentation(image_array: np.ndarray, segmentation: Segmentation, draw_boxes: bool = True, alpha: float = 0.5) -> np.ndarray:
    annotated_img = image_array.copy()
    
    for label, polygon, box in zip(segmentation.labels, segmentation.polygons, segmentation.boxes):
        color = (255, 0, 0) if label == "danger" else (0, 255, 0)  

        pts = np.array(polygon, np.int32).reshape((-1, 1, 2))

        cv2.fillPoly(annotated_img, [pts], color)

        overlay = image_array.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, alpha, annotated_img, 1 - alpha, 0, annotated_img)

        if draw_boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated_img,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2,
            )
    
    return annotated_img

class GunDetector:
    def __init__(self) -> None:
        print(f"Loading object detection model: {SETTINGS.od_model_path}")
        self.od_model = YOLO(SETTINGS.od_model_path)
        print(f"Loading segmentation model: {SETTINGS.seg_model_path}")
        self.seg_model = YOLO(SETTINGS.seg_model_path)

    def detect_guns(self, image_array: np.ndarray, threshold: float = 0.5) -> Detection:
        results = self.od_model(image_array, conf=threshold)[0]
        labels = results.boxes.cls.tolist()
        indexes = [i for i in range(len(labels)) if labels[i] in [3, 4]] 

        boxes = [[int(v) for v in box] for i, box in enumerate(results.boxes.xyxy.tolist()) if i in indexes]
        confidences = [c for i, c in enumerate(results.boxes.conf.tolist()) if i in indexes]
        labels_txt = [results.names[labels[i]] for i in indexes]
        
        return Detection(
            pred_type=PredictionType.object_detection,
            n_detections=len(boxes),
            boxes=boxes,
            labels=labels_txt,
            confidences=confidences,
        )

    def segment_people(self, image_array: np.ndarray, threshold: float = 0.5, max_distance: int = 10) -> Segmentation:
        results = self.seg_model(image_array, conf=threshold)[0]
        labels = results.boxes.cls.tolist()

        person_indexes = [i for i in range(len(labels)) if labels[i] == 0]  # Detectar personas
        person_boxes = [[int(v) for v in box] for i, box in enumerate(results.boxes.xyxy.tolist()) if i in person_indexes]
        person_polygons = [[[int(coord[0]), int(coord[1])] for coord in results.masks.xy[i]] for i in person_indexes]

        gun_detections = self.detect_guns(image_array, threshold)

        person_labels = []
        for person_box, person_polygon in zip(person_boxes, person_polygons):
            matched_gun = match_gun_bbox(person_polygon, gun_detections.boxes, max_distance)
            person_labels.append("danger" if matched_gun else "safe")

        return Segmentation(
            pred_type=PredictionType.segmentation,
            n_detections=len(person_boxes),
            polygons=person_polygons,
            boxes=person_boxes,
            labels=person_labels
        )
