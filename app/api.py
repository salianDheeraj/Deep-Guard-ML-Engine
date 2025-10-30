import os
import uuid
import shutil
from fastapi import APIRouter, UploadFile, File, BackgroundTasks
from app.video_utils import extract_frames
# from app.model import detect_deepfake
from app.schemas import PredictionResponse
from app.config.config import ExtractionConfig
from fastapi.responses import FileResponse
from app.services.save_video import VideoSaver
from app.services.video_preprocessor import VideoPreprocessor
from fastapi import HTTPException
from app.utils.delayed_cleanup import delayed_cleanup

router = APIRouter()

@router.post("/detect/deepfake/video", response_class=FileResponse)
async def predict_video(file: UploadFile = File(...), frames: int = 50, background_tasks: BackgroundTasks = None):
    # Save uploaded video to temp dir
    # VideoSaver(file from user, temp directory path, unique video id generated)
    video_path, video_folder = VideoSaver.save_file(file, ExtractionConfig.TEMP_DIR, str(uuid.uuid4()))

    # Extract frames
    preprocessor = VideoPreprocessor(ExtractionConfig)
    stats = preprocessor.preprocess_frame(
        video_path=video_path,
        output_dir=video_folder,
        video_id=os.path.basename(video_folder),
        frames=frames
    )

    #if there are preprocessing errors, raise exception
    if stats.errors:
        raise HTTPException(status_code=400, detail={
            "video_id": stats.video_id,
            "message": "Video preprocessing failed",
            "errors": stats.errors
        })
    

    # Detect fake
    # result, fake_frames = detect_deepfake(frames)

    # 3. Schedule file cleanup AFTER response is sent
    background_tasks.add_task(delayed_cleanup, video_folder, delay =60)

    return FileResponse(
        path=video_path,
        media_type='video/mp4',
        filename=file.filename,
    )
