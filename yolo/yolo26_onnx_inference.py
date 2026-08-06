import ast
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

MODEL_PATH = Path(r"/Users/autumn/IdeaProjects/ultralytics/ultralytics/yolo_workspace/model_export/onnx/yolo26s.onnx")
IMAGE_PATH = Path(r"/Users/autumn/IdeaProjects/ultralytics/ultralytics/assets/bus.jpg")
OUTPUT_PATH = Path(r"/Users/autumn/IdeaProjects/ultralytics/ultralytics/yolo_workspace/test_output/result.jpg")
CONFIDENCE = 0.25
IMAGE_SIZE = 640  # Only used when the ONNX model has dynamic input dimensions.
DEVICE = "cpu"  # Use "cuda" after installing onnxruntime-gpu.


def letterbox(image, new_shape):
    """Resize and pad an image while preserving its aspect ratio."""
    height, width = image.shape[:2]
    new_height, new_width = new_shape
    gain = min(new_height / height, new_width / width)
    resized_width, resized_height = round(width * gain), round(height * gain)
    pad_width = (new_width - resized_width) / 2
    pad_height = (new_height - resized_height) / 2

    if (width, height) != (resized_width, resized_height):
        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

    top, bottom = round(pad_height - 0.1), round(pad_height + 0.1)
    left, right = round(pad_width - 0.1), round(pad_width + 0.1)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return image, gain, (left, top)


def load_names(session):
    """Read class names embedded by the Ultralytics ONNX exporter."""
    names = session.get_modelmeta().custom_metadata_map.get("names")
    return ast.literal_eval(names) if names else {}


def create_session(model_path, device):
    """Create an ONNX Runtime session on the requested device."""
    available = ort.get_available_providers()
    if device == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError("CUDAExecutionProvider is unavailable. Install onnxruntime-gpu or use --device cpu.")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(model_path, providers=providers)


def draw_detection(image, box, score, class_id, class_name):
    """Draw one detection on the image."""
    x1, y1, x2, y2 = box.astype(int)
    color = ((37 * class_id) % 255, (17 * class_id + 79) % 255, (29 * class_id + 149) % 255)
    label = f"{class_name} {score:.2f}"
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    text_y = max(y1, text_height + baseline)
    cv2.rectangle(
        image,
        (x1, text_y - text_height - baseline),
        (x1 + text_width, text_y),
        color,
        cv2.FILLED,
    )
    cv2.putText(image, label, (x1, text_y - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def main():
    """Run YOLO26 ONNX inference and save the annotated image."""
    session = create_session(str(MODEL_PATH), DEVICE)
    model_input = session.get_inputs()[0]

    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {IMAGE_PATH}")

    _, _, input_height, input_width = model_input.shape
    input_height = input_height if isinstance(input_height, int) else IMAGE_SIZE
    input_width = input_width if isinstance(input_width, int) else IMAGE_SIZE
    input_image, gain, (pad_x, pad_y) = letterbox(image, (input_height, input_width))
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = input_image.transpose(2, 0, 1)[None] / 255.0
    input_dtype = np.float16 if model_input.type == "tensor(float16)" else np.float32
    input_image = np.ascontiguousarray(input_image, dtype=input_dtype)

    predictions = session.run(None, {model_input.name: input_image})[0][0]
    if predictions.ndim != 2 or predictions.shape[1] != 6:
        raise ValueError(
            f"Expected YOLO26 end-to-end output shaped (N, 6), but received {predictions.shape}. "
            "Export the model with end2end=True."
        )

    predictions = predictions[predictions[:, 4] >= CONFIDENCE]
    names = load_names(session)
    boxes = predictions[:, :4].copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / gain
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / gain
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, image.shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, image.shape[0])

    for box, prediction in zip(boxes, predictions):
        score, class_id = float(prediction[4]), int(prediction[5])
        class_name = names.get(class_id, str(class_id))
        draw_detection(image, box, score, class_id, class_name)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(OUTPUT_PATH), image):
        raise RuntimeError(f"Unable to save result: {OUTPUT_PATH}")
    print(f"Detected {len(predictions)} objects. Result saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
