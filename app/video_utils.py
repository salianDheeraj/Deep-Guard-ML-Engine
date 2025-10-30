import cv2

def extract_frames(video_path, every_n_frames=50):
    """Extract every nth frame from a video."""
    vidcap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    while True:
        success, frame = vidcap.read()
        if not success:
            break
        if count % every_n_frames == 0:
            frames.append(frame)
        count += 1

    vidcap.release()
    return frames
