# import numpy as np
# import tensorflow as tf
# import cv2
# from app.config import MODEL_PATH, FAKE_THRESHOLD

# # Load model once on startup
# model = tf.keras.models.load_model(MODEL_PATH)

# def detect_deepfake(frames):
#     """Run deepfake detection on extracted frames."""
#     fake_indices = []

#     for i, frame in enumerate(frames):
#         inp = preprocess_frame(frame)
#         pred = model.predict(inp, verbose=0)[0][0]  # Assuming single output neuron
#         if pred > FAKE_THRESHOLD:
#             fake_indices.append(i)

#     result = "fake" if fake_indices else "real"
#     return result, fake_indices


# def preprocess_frame(frame):
#     """Resize, normalize and reshape frame for model input."""
#     frame = cv2.resize(frame, (299, 299))       # match your model’s input size
#     frame = frame.astype("float32") / 255.0     # normalize
#     frame = np.expand_dims(frame, axis=0)       # add batch dimension
#     return frame
