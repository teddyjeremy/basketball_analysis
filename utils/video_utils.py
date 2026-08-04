from pathlib import Path

import cv2


def get_video_info(video_path):
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    if fps <= 0:
        raise ValueError(f"Invalid video FPS: {fps}")

    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": frame_count / fps,
    }


def read_video(video_path, start_frame=0, end_frame=None, stride=1):
    if stride < 1:
        raise ValueError("stride must be at least 1")

    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_frame is None:
        end_frame = total_frames

    if start_frame < 0 or end_frame < start_frame:
        cap.release()
        raise ValueError("Invalid frame range")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    frame_index = start_frame

    while frame_index < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if (frame_index - start_frame) % stride == 0:
            frames.append(frame)

        frame_index += 1

    cap.release()
    return frames


def save_video(output_video_frames, output_video_path, fps=30.0, codec="mp4v"):
    if not output_video_frames:
        raise ValueError("No video frames were provided")

    output_path = Path(output_video_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    first_frame = output_video_frames[0]
    if first_frame is None or first_frame.ndim != 3:
        raise ValueError("Video frames must be color images")

    height, width = first_frame.shape[:2]
    if height == 0 or width == 0:
        raise ValueError("Video frame has invalid dimensions")

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise ValueError(f"Unable to create output video: {output_path}")

    try:
        for frame in output_video_frames:
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
    finally:
        writer.release()
