import onnxruntime as ort
import numpy as np
import cv2
import time
import os


class ONNXInference:
    def __init__(self, model_file_path):
        # Ensure the file exists
        if not os.path.exists(model_file_path):
            raise FileNotFoundError(f"ONNX model file not found: {model_file_path}")

        # Load ONNX Runtime session
        self.session = ort.InferenceSession(model_file_path, providers=["CUDAExecutionProvider"])

        # Get input and output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Get input shape
        self.input_shape = self.session.get_inputs()[0].shape
        print(f"ONNX model loaded successfully: {model_file_path}")
        print(f"Input shape: {self.input_shape}, Output name: {self.output_name}")

    def preprocess(self, frame):
        # Resize and normalize the input frame
        input_image = cv2.resize(frame, (self.input_shape[2], self.input_shape[3]))
        input_image = input_image.astype(np.float32) / 255.0  # Normalize to [0, 1]
        input_image = np.transpose(input_image, (2, 0, 1))  # HWC to CHW
        input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension
        return input_image

    def infer(self, frame):
        # Preprocess the frame
        try:
            preprocessed = self.preprocess(frame)

            # Run inference
            output = self.session.run([self.output_name], {self.input_name: preprocessed})

            return output[0]
        except Exception as e:
            print(f"Inference failed: {e}")
            return None


def postprocess_and_draw(frame, output):
    """
    Postprocess the output and draw results on the frame.
    This step depends on the model's output format.
    """
    if output is None:
        return frame

    # Example: Assuming output contains bounding boxes and confidence scores
    for detection in output:
        try:
            x1, y1, x2, y2, confidence, class_id = detection  # Adjust based on your model's output format
            if confidence > 0.5:  # Threshold for confidence
                label = f"Class {int(class_id)}: {confidence:.2f}"
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        except Exception as e:
            print(f"Error in postprocessing: {e}")
    return frame


if __name__ == "__main__":
    # Initialize ONNX inference
    model_path = "v11m_trained.onnx"
    try:
        onnx_infer = ONNXInference(model_path)
    except Exception as e:
        print(f"Failed to initialize ONNXInference: {e}")
        exit()

    # Load video
    video_path = "UAV.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        exit()

    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter("output.avi", fourcc, 30, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference
        start_time = time.time()
        output = onnx_infer.infer(frame)
        end_time = time.time()

        # Postprocess and draw
        processed_frame = postprocess_and_draw(frame, output)

        # Display FPS on the frame
        fps = 1 / (end_time - start_time)
        cv2.putText(processed_frame, f"FPS: {fps:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Write frame to output video
        out.write(processed_frame)

        # Display the frame
        cv2.imshow("ONNX Inference", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
