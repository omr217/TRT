# 🚀 ONNX Runtime Inference for Object Detection

This project demonstrates **real-time object detection** using a pretrained **ONNX model** with **ONNX Runtime (CUDA)**. It reads a video stream, performs inference, and displays bounding boxes, class labels, and FPS on each frame.

---

## 🧰 Features

- ✅ Loads a YOLO-style ONNX model using `onnxruntime` with GPU acceleration.
- ✅ Performs preprocessing (resize, normalize, transpose).
- ✅ Runs model inference and postprocessing to draw bounding boxes.
- ✅ Displays **FPS** in real time and saves processed output to a video file.
- ✅ Modular design with error handling.

---

## 📂 Project Structure

.
├── v11m_trained.onnx # Your ONNX model file
├── UAV.mp4 # Input video file
├── output.avi # Output video with detections
├── onnx_inference.py # Main inference script (this code)
└── README.md

---

## 📦 Requirements

Install the necessary packages:

```bash
pip install onnxruntime-gpu numpy opencv-python
```

Make sure:

Your machine has a supported CUDA GPU for onnxruntime-gpu.

Your ONNX model is compatible with the input/output assumptions in this script.


### 🎯 Model Assumptions
The script assumes the ONNX model output is in the format:

python 
```
[x1, y1, x2, y2, confidence, class_id]
```

#### 🧪 Usage
Place your ONNX model as v11m_trained.onnx.

Add an input video (e.g., UAV.mp4).

Run the script:
```
python onnx_inference.py
```

##### ⚠️ Troubleshooting
ModelNotFoundError: Check your ONNX model path.

CUDAExecutionProvider error: Ensure your GPU and drivers are correctly installed.

Wrong shape errors: Check model input shape (e.g., [1, 3, 640, 640]) and adjust preprocessing accordingly.






