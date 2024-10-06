from fastapi import FastAPI, UploadFile, File, HTTPException, status, Depends
from fastapi.responses import Response
from src.predictor import GunDetector, Detection, Segmentation, annotate_detection, annotate_segmentation
from src.models import Gun, Person, PixelLocation  
from src.config import get_settings
import io
from PIL import Image, UnidentifiedImageError
import numpy as np
from functools import cache
import cv2

SETTINGS = get_settings()
app = FastAPI(title=SETTINGS.api_name, version=SETTINGS.revision)


@cache
def get_gun_detector() -> GunDetector:
    print("Creating model...")
    return GunDetector()


def load_image(file: UploadFile) -> np.ndarray:
    img_stream = io.BytesIO(file.file.read())
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Not an image"
        )
    try:
        img_obj = Image.open(img_stream)
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Image format not supported"
        )
    img_array = np.array(img_obj)
    return img_array

def detect_uploadfile(detector: GunDetector, file: UploadFile, threshold: float) -> tuple[Detection, np.ndarray]:
    img_array = load_image(file)
    return detector.detect_guns(img_array, threshold), img_array

@app.get("/model_info")
def get_model_info(detector: GunDetector = Depends(get_gun_detector)):
    return {
        "model_name": "Gun detector",
        "gun_detector_model": detector.od_model.model.__class__.__name__,
        "semantic_segmentation_model": detector.seg_model.model.__class__.__name__,
        "input_type": "image",
    }


@app.post("/detect_guns")
def detect_guns(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> Detection:
    results, _ = detect_uploadfile(detector, file, threshold)
    return results


@app.post("/annotate_guns")
def annotate_guns(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> Response:
    detection, img = detect_uploadfile(detector, file, threshold)
    annotated_img = annotate_detection(img, detection)

    img_pil = Image.fromarray(annotated_img)
    image_stream = io.BytesIO()
    img_pil.save(image_stream, format="JPEG")
    image_stream.seek(0)
    return Response(content=image_stream.read(), media_type="image/jpeg")

@app.post("/detect_people", response_model=Segmentation)
def detect_people(
    threshold: float = 0.5,
    max_distance: int = 200,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector)
) -> Segmentation:
    img_array = load_image(file)
    segmentation = detector.segment_people(img_array, threshold=threshold, max_distance=max_distance)
    return segmentation


@app.post("/annotate_people")
def annotate_people(
    threshold: float = 0.5,
    max_distance: int = 200,
    draw_boxes: bool = True,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector)
) -> Response:
    img_array = load_image(file)
    segmentation = detector.segment_people(img_array, threshold=threshold, max_distance=max_distance)
    annotated_img = annotate_segmentation(img_array, segmentation, draw_boxes=draw_boxes)
    
    img_pil = Image.fromarray(annotated_img)
    image_stream = io.BytesIO()
    img_pil.save(image_stream, format="JPEG")
    image_stream.seek(0)
    return Response(content=image_stream.read(), media_type="image/jpeg")


@app.post("/detect", response_model=dict)
def detect(
    threshold: float = 0.5,
    max_distance: int = 200,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector)
) -> dict:
    img_array = load_image(file)
    detection = detector.detect_guns(img_array, threshold=threshold)
    segmentation = detector.segment_people(img_array, threshold=threshold, max_distance=max_distance)
    
    return {
        "detection": detection,
        "segmentation": segmentation
    }


@app.post("/annotate")
def annotate(
    threshold: float = 0.5,
    max_distance: int = 200,
    draw_boxes: bool = True,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector)
) -> Response:
    img_array = load_image(file)
    detection = detector.detect_guns(img_array, threshold=threshold)
    segmentation = detector.segment_people(img_array, threshold=threshold, max_distance=max_distance)
    
    annotated_img = annotate_detection(img_array, detection)
    annotated_img = annotate_segmentation(annotated_img, segmentation, draw_boxes=draw_boxes)
    
    img_pil = Image.fromarray(annotated_img)
    image_stream = io.BytesIO()
    img_pil.save(image_stream, format="JPEG")
    image_stream.seek(0)
    return Response(content=image_stream.read(), media_type="image/jpeg")


@app.post("/guns", response_model=list[Gun])
def guns(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector)
) -> list[Gun]:
    img_array = load_image(file)
    detection = detector.detect_guns(img_array, threshold=threshold)
    
    guns = []
    for box, label in zip(detection.boxes, detection.labels):
        x_center = (box[0] + box[2]) // 2
        y_center = (box[1] + box[3]) // 2
        gun_type = "pistol" if label == "pistol" else "rifle"
        gun = Gun(gun_type=gun_type, location=PixelLocation(x=x_center, y=y_center))
        guns.append(gun)
    
    return guns


@app.post("/people", response_model=list[Person])
def people(
    threshold: float = 0.5,
    max_distance: int = 200,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector)
) -> list[Person]:
    img_array = load_image(file)
    segmentation = detector.segment_people(img_array, threshold=threshold, max_distance=max_distance)
    
    people = []
    for polygon, label in zip(segmentation.polygons, segmentation.labels):
        poly = np.array(polygon)
        x_center = int(np.mean(poly[:, 0]))
        y_center = int(np.mean(poly[:, 1]))
        area = int(cv2.contourArea(poly))
        
        person = Person(
            person_type=label,
            location=PixelLocation(x=x_center, y=y_center),
            area=area
        )
        people.append(person)
    
    return people

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app:app", port=8080, host="0.0.0.0")
