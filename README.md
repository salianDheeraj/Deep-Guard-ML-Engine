# Deep Guard ML Engine

The **Deep Guard ML Engine** is a high-performance, FastAPI-based microservice designed for detecting deepfakes in images and videos. It utilizes a **TensorFlow Lite (TFLite)** model for efficient inference and **OpenCV** for advanced face tracking and extraction.

> **For a detailed technical overview, please refer to the [System Architecture](ARCHITECTURE.md).**

## 🚀 Key Features

-   **Deepfake Detection:** Analyzes both singular images and video frames to determine authenticity (Real vs. Fake).
-   **Optimized Video Processing:** Implements sequential frame reading and pre-allocated buffers for 20-30% faster processing.
-   **Batch Image Analysis:** Supports bulk uploading and processing of images.
-   **Smart Face Tracking:** Uses a 3D Face Tracker to ensure consistent face cropping across video frames.
-   **Automated Cleanup:** Background tasks automatically clean up temporary files after processing to manage disk space.
-   **Annotated Reports:** Generates comprehensive ZIP reports containing annotated frames/images and JSON confidence logs.

## 🛠️ Tech Stack

| Technology | Purpose |
| :--- | :--- |
| **Python 3.10+** | Core programming language. |
| **FastAPI** | High-performance async web framework for the API. |
| **TensorFlow Lite** | Lightweight, optimized inference engine for the deepfake model. |
| **OpenCV (cv2)** | Computer vision tasks: video reading, face detection, and image manipulation. |
| **NumPy** | High-speed numerical operations for tensor manipulation. |
| **Uvicorn** | ASGI server for running the FastAPI application. |

## 📂 Project Structure

```
Deep-Guard-ML-Engine/
├── app/
│   ├── main.py              # Application entry point
│   ├── config/              # Configuration settings (constants, paths)
│   ├── routes/              # API Endpoints
│   │   ├── video_detection.py # /detect/deepfake/video logic
│   │   └── image_detection.py # /detect/deepfake/images logic
│   ├── services/            # Business Logic Services
│   │   ├── model.py         # TFLite Inference wrapper
│   │   ├── *_preprocessor.py# Image/Video orchestration
│   │   └── *_saver.py       # File I/O handlers
│   └── utils/               # Core Utilities
│       ├── face_tracker.py  # 3D Face Detection & Tracking
│       ├── face_extractor.py# Conservative cropping logic
│       └── video_processor.py# Optimized video frame reader
├── models/                  # ML Artifacts (TFLite models)
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

## ⚡ API Endpoints

### 1. Detect Video Deepfake
**POST** `/detect/deepfake/video`

Analyzes a video file, extracts faces, runs inference, and returns a ZIP report.

-   **Form Data:**
    -   `file`: The video file (`.mp4`, `.avi`, etc).
    -   `frames` (optional, default 50): Number of frames to extract and analyze.
-   **Response:** `application/zip` containing annotated frames and `confidence_report.json`.
-   **Headers:** `X-Average-Confidence`, `X-Video-ID`.

### 2. Detect Image Deepfakes (Batch)
**POST** `/detect/deepfake/images`

Analyzes a batch of uploaded images.

-   **Form Data:**
    -   `files`: List of image files (`.jpg`, `.png`).
-   **Response:** `application/zip` containing annotated images and report.

## ⚙️ Installation & Setup

### Prerequisites
-   Python 3.8 - 3.11 (Recommended)
-   **FFmpeg** installed system-wide (for video processing).

### Steps

1.  **Clone the Repository**
    ```bash
    git clone <repo-url>
    cd Deep-Guard-ML-Engine
    ```

2.  **Create Virtual Environment**
    ```bash
    python -m venv .venv
    # Windows:
    .venv\Scripts\activate
    # Mac/Linux:
    source .venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Server**
    ```bash
    uvicorn app.main:app --reload --port 8000
    ```

### Docker

The repository includes a Dockerfile for the ML engine, so you can build and run it directly with Docker.

1.  Build the image:
    ```bash
    docker build -t deep-guard-ml-engine .
    ```

2.  Run the container:
    ```bash
    docker run --rm -p 8000:8000 deep-guard-ml-engine
    ```

3.  Open the API docs at `http://localhost:8000/docs` after the container starts.

The API will be available at `http://localhost:8000`. Documentation is available at `http://localhost:8000/docs`.