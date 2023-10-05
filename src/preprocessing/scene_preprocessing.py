from typing import Tuple

import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
from multiprocessing import Process

sys.path.append('../')
from utils.video_utils import get_video, get_frame_mask, apply_frame_mask

DATA_PATH = '../../data/'
PREPROCESSING_DATA_PATH = DATA_PATH + 'preprocessing/'

RAW_SCENES_PATH = PREPROCESSING_DATA_PATH + 'raw_scenes/'
PROCESSED_SCENES_PATH = PREPROCESSING_DATA_PATH + 'processed_scenes/'
PROCESSED_REFRAMED_SCENES_PATH = PREPROCESSING_DATA_PATH + 'processed_reframed_scenes/'

def mask_reframe_frame(frame: np.ndarray, type: str) -> np.ndarray:
    """
    Applies the left or right mask to the given frame and reframes it.

    Args:
        frame (np.ndarray): Frame to be masked and reframed.
        type (str): Type of mask to be applied.

    Returns:
        masked_reframed_frame (np.ndarray): Masked and reframed frame.
    """

    assert type in ['left', 'right'], f"❌ Invalid mask type {type}: must be 'left' or 'right'."

    # Mask
    masked_frame = apply_frame_mask(frame, type)

    # Reframe
    masked_reframed_frame = masked_frame[~np.all(masked_frame == 0, axis=2).all(axis=1)]
    masked_reframed_frame = masked_reframed_frame[:, ~np.all(masked_reframed_frame == 0, axis=2).all(axis=0)]

    return masked_reframed_frame

def preprocess_frame(
        frame: np.ndarray, 
        clahe_clip_limit: float = 3.0, 
        clahe_tile_grid_size: Tuple[int, int] = (10, 10),
        bilateral_d: int = 13,
        bilateral_sigma_color: int = 15,
        bilateral_sigma_space: int = 15,
        mask: np.ndarray = None
    ) -> np.ndarray:
    """
    Preprocesses the given frame.
    
    Args:
        frame (np.ndarray): Frame to be preprocessed.
        clahe_clip_limit (float): Clip limit for CLAHE.
        clahe_tile_grid_size (Tuple[int, int]): Tile grid size for CLAHE.
        bilateral_d (int): Diameter of each pixel neighborhood for bilateral filtering.
        bilateral_sigma_color (int): Filter sigma in the color space for bilateral filtering.
        bilateral_sigma_space (int): Filter sigma in the coordinate space for bilateral filtering.
        mask (np.ndarray): Mask to be applied to the frame.
        
    Returns:
        preprocessed_frame (np.ndarray): Preprocessed frame.
    """

    frame = frame.astype(np.uint8)

    # Apply CLAHE for illumination correction
    clahe_frame = frame.copy()
    clahe_frame = cv2.cvtColor(clahe_frame, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid_size)
    clahe_frame[:, :, 0] = clahe.apply(clahe_frame[:, :, 0])
    clahe_frame = cv2.cvtColor(clahe_frame, cv2.COLOR_LAB2BGR)

    # Only apply CLAHE to masked areas
    if mask is not None:
        frame = np.where(mask[:, :, np.newaxis] == 1, clahe_frame, frame)

    # Apply Bilateral Filtering for noise reduction on the entire frame
    frame = cv2.bilateralFilter(frame, d=bilateral_d, sigmaColor=bilateral_sigma_color, sigmaSpace=bilateral_sigma_space)

    return frame.astype(np.uint8)

def process_frames(
        output_video_path: str,
        input_video_path: str, 
        start_frame: int, 
        end_frame: int, 
        scene_file: str, 
        mask_reframe: bool = False,
        n_frames_to_average: int = 3
    ) -> None:
    """
    Processes the frames of the given video.
    
    Args:
        output_video_path (str): Path to the output video.
        input_video_path (str): Path to the input video.
        start_frame (int): Start frame of the video.
        end_frame (int): End frame of the video.
        scene_file (str): Name of the scene file.
        mask_reframe (bool): Whether to mask and reframe the frames.
        n_frames_to_average (int): Number of frames to average for temporal averaging.
    """

    input_video = get_video(input_video_path)
    input_video.set(cv2.CAP_PROP_POS_FRAMES, start_frame) # Set start frame

    output_video = None
    frame_buffer = [] # Buffer for averaging frames with temporal averaging
    i = start_frame

    if start_frame == 0:
        bar = tqdm(total=end_frame - start_frame, desc=f'▶️ Processing {scene_file}')
    while i < end_frame:
        ret, frame = input_video.read()
        if not ret:
            break

        # Process left frame
        left_frame, _ = frame[:, :frame.shape[1]//2], frame[:, frame.shape[1]//2:]
        if mask_reframe:
            left_frame = mask_reframe_frame(left_frame, 'left')
        mask = get_frame_mask('left', reframe=mask_reframe)
        preprocessed_frame = preprocess_frame(left_frame, mask=mask)

        # Apply temporal averaging
        frame_buffer.append(preprocessed_frame)
        preprocessed_frame = np.mean(frame_buffer, axis=0).astype(np.uint8)
        if len(frame_buffer) == n_frames_to_average:
            frame_buffer.pop(0) # Remove oldest frame from buffer

        # Make sure that the output video is initialized before writing frames
        if output_video is None:
            frame_width = preprocessed_frame.shape[1]
            frame_height = preprocessed_frame.shape[0]
            frame_rate = int(input_video.get(5))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_video = cv2.VideoWriter(
                os.path.join(output_video_path, f'{str(start_frame)}-{str(end_frame)}_{scene_file}'),
                fourcc,
                frame_rate,
                (frame_width, frame_height)
            )

        # Write frame to output video
        output_video.write(preprocessed_frame)
        i += 1
        if start_frame == 0:
            bar.update(1)

    input_video.release()

def process_video(
        output_video_path: str, 
        scene_file: str, 
        num_processes: int = 4, 
        mask_reframe: bool = False):
    """
    Processes the given video.

    Args:
        output_video_path (str): Path to the output video.
        scene_file (str): Name of the scene file.
        num_processes (int): Number of processes to use.
        mask_reframe (bool): Whether to mask and reframe the frames.
    """

    print(f'▶️ Processing {output_video_path}{scene_file}...')

    # Get number of frames in video
    input_video_path = os.path.join(RAW_SCENES_PATH, scene_file)
    input_video = get_video(input_video_path)
    frame_count = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
    input_video.release()

    # Create separate processes for processing frames
    frames_per_process = frame_count // num_processes
    read_processes = []
    for i in range(num_processes):
        start_frame = i * frames_per_process
        end_frame = start_frame + frames_per_process if i < num_processes - 1 else frame_count
        read_process = Process(target=process_frames, args=(output_video_path, input_video_path, start_frame, end_frame, scene_file, mask_reframe))
        read_processes.append(read_process)
        read_process.start()

    # Wait for all processes to finish
    for read_process in read_processes:
        read_process.join()

    # Get all processed videos
    processed_videos = os.listdir(output_video_path)
    processed_videos = [video for video in processed_videos if video.endswith(scene_file)]

    # Sort processed videos by start frame
    processed_videos.sort(key=lambda video: int(video.split('_')[0].split('-')[0]))

    # Combine processed videos
    print(f'▶️ Combining {len(processed_videos)} processed videos...')
    output_video = None
    for processed_video in tqdm(processed_videos):
        processed_video_path = os.path.join(output_video_path, processed_video)
        processed_video = get_video(processed_video_path)

        while True:
            ret, frame = processed_video.read()
            if not ret:
                break

            # Make sure that the output video is initialized before writing frames
            if output_video is None:
                frame_width = frame.shape[1]
                frame_height = frame.shape[0]
                frame_rate = int(processed_video.get(5))
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                output_video = cv2.VideoWriter(
                    os.path.join(output_video_path, scene_file),
                    fourcc,
                    frame_rate,
                    (frame_width, frame_height)
                )

            # Write frame to output video
            output_video.write(frame)

        processed_video.release()

    output_video.release()

    # Delete processed videos
    for processed_video in processed_videos:
        os.remove(os.path.join(output_video_path, processed_video))

    print(f'✅ Processed {output_video_path}{scene_file}.')

def ask_n_processes():
    """
    Asks the user for the number of processes to use.
    """

    cpu_count = os.cpu_count()
    num_processes = input(f"Enter number of processes (default {cpu_count}): ")
    if num_processes == '':
        num_processes = cpu_count
    else:
        try:
            num_processes = int(num_processes)
            if num_processes <= 0 or num_processes > cpu_count:
                raise Exception()
        except:
            print(f"❌ Invalid number of processes: must be a positive integer less than or equal to the number of CPU cores available ({cpu_count}).")
            exit()

    return num_processes

def ask_mask_reframe():
    """
    Asks the user whether to mask and reframe the frames.
    """

    mask_reframe = input("Mask and reframe frames? (y/n): ")
    mask_reframe = mask_reframe.lower() == 'y'

    return mask_reframe

if __name__ == "__main__":
    # Ask user for number of processes and whether to mask and reframe frames
    num_processes = ask_n_processes()
    mask_reframe = ask_mask_reframe()

    print(f"▶️ Processing {'and reframing ' if mask_reframe else ''}scenes using {num_processes} processes...")

    # Create output video directory
    output_video_path = PROCESSED_REFRAMED_SCENES_PATH if mask_reframe else PROCESSED_SCENES_PATH
    if not os.path.exists(output_video_path):
        os.makedirs(output_video_path)

    # Process videos
    if not os.path.exists(RAW_SCENES_PATH):
        print(f"❌ Invalid raw scenes path {RAW_SCENES_PATH}: directory does not exist.")
        exit()

    scene_files = os.listdir(RAW_SCENES_PATH)
    for scene_file in scene_files:
        process_video(output_video_path, scene_file, num_processes=num_processes, mask_reframe=mask_reframe)