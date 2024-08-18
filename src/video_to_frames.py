import cv2
import os
import shutil

def video_to_frames(source, fps=None, force_rebuild=False):
    """
    Extract frames from a video file and save them in a new folder next to the video.

    :param source: Path to the source video file
    :param fps: Frames per second to extract. If None, extract all frames
    :param force_rebuild: If True, rebuild the frames folder even if it already exists
    :return: Tuple containing number of frames extracted and path to the frames folder
    """
    source = os.path.normpath(source)
    video_name = os.path.splitext(os.path.basename(source))[0]
    frames_folder = os.path.join(os.path.dirname(source), f"{video_name}_frames")

    if os.path.exists(frames_folder) and not force_rebuild:
        print(f"Frames folder already exists at {frames_folder}")
        frame_count = len([f for f in os.listdir(frames_folder) if f.endswith('.jpg')])
        return frame_count, frames_folder

    if os.path.exists(frames_folder):
        shutil.rmtree(frames_folder)
    os.makedirs(frames_folder)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {source}")

    if fps is None:
        fps = int(cap.get(cv2.CAP_PROP_FPS))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % (int(cap.get(cv2.CAP_PROP_FPS)) // fps) == 0:
            frame_path = os.path.join(frames_folder, f'frame_{frame_count:04d}.jpg')
            cv2.imwrite(frame_path, frame)

        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames to {frames_folder}")
    return frame_count, frames_folder