# Ensure that the current working directory is this file
from pathlib import Path
CURRENT_DIR = Path(__file__).parent

import os
import cv2
import numpy as np
from pathlib import Path

DATA_PATH = str(CURRENT_DIR / '..' / '..' / 'data') + '/'

LEFT_MASK_PATH = DATA_PATH + 'left_mask.png'
RIGHT_MASK_PATH = DATA_PATH + 'right_mask.png'

def get_video(video_path: str) -> cv2.VideoCapture:
    """
    Returns a cv2.VideoCapture object for the video at the given path.

    Args:
        video_path (str): Path to the video.

    Returns:
        cv2.VideoCapture: VideoCapture object for the video.
    """

    assert os.path.exists(video_path), f"❌ Invalid video path {video_path}: file does not exist."

    return cv2.VideoCapture(video_path)

def get_frame_mask(type: str, reframe: bool = False) -> np.ndarray:
    """
    Returns the left or right mask, optionally reframed.

    Args:
        type (str): Type of mask to be returned.
        reframe (bool): Whether to reframe the mask.

    Returns:
        mask (np.ndarray): Left or right mask, with values in [0, 1].
    """
    
    assert type in ['left', 'right'], f"❌ Invalid mask type {type}: must be 'left' or 'right'."

    mask = cv2.imread(LEFT_MASK_PATH, cv2.IMREAD_GRAYSCALE) if type == 'left' else cv2.imread(RIGHT_MASK_PATH, cv2.IMREAD_GRAYSCALE)
    mask = np.where(mask > 255 * 0.5, 1, 0)

    if reframe:
        # Remove rows containing all zeros
        mask = mask[~np.all(mask == 0, axis=1)]

        # Remove columns containing all zeros
        mask = mask[:, ~np.all(mask == 0, axis=0)]

    return mask.astype(np.uint8)

def apply_frame_mask(frame: np.ndarray, type: str) -> np.ndarray:
    """
    Applies the left or right mask to the given frame.

    Args:
        frame (np.ndarray): Frame to be masked.
        type (str): Type of mask to be applied.

    Returns:
        masked_frame (np.ndarray): Masked frame.
    """
    assert type in ['left', 'right'], f"❌ Invalid mask type {type}: must be 'left' or 'right'."

    mask = get_frame_mask(type)
    masked_frame = frame * mask[:, :, np.newaxis]

    return masked_frame

def get_frame_from_video(video: cv2.VideoCapture, frame: int, split: bool = True, masked: bool = False, reframe: bool = False) -> np.ndarray:
    """
    Returns left and right frames at the given index from the video, optionally masked and reframed.

    Args:
        video (cv2.VideoCapture): VideoCapture object for the video.
        frame (int): Index of the frame to be returned.
        split (bool): Whether to return the left and right frames separately.
        masked (bool): Whether to return the masked version of the frame.
        reframe (bool): Whether to reframe the frame.

    Returns:
        left_frame (np.ndarray): Left frame at the given index, with values in [0, 255].
        right_frame (np.ndarray): Right frame at the given index, with values in [0, 255].
    """
    assert frame >= 0 and frame < video.get(cv2.CAP_PROP_FRAME_COUNT), f"❌ Invalid frame index {frame}: video has {int(video.get(cv2.CAP_PROP_FRAME_COUNT))} frames."

    video.set(cv2.CAP_PROP_POS_FRAMES, frame)
    _, frame = video.read()

    if split:
        left_frame = frame[:, :frame.shape[1]//2]
        right_frame = frame[:, frame.shape[1]//2:]
    else:
        left_frame = frame
        right_frame = frame

    if masked:
        left_frame = apply_frame_mask(left_frame, 'left')
        right_frame = apply_frame_mask(right_frame, 'right')

    if reframe:
        # Remove rows containing all zeros
        left_frame = left_frame[~np.all(left_frame == 0, axis=2).all(axis=1)]
        right_frame = right_frame[~np.all(right_frame == 0, axis=2).all(axis=1)]

        # Remove columns containing all zeros
        left_frame = left_frame[:, ~np.all(left_frame == 0, axis=2).all(axis=0)]
        right_frame = right_frame[:, ~np.all(right_frame == 0, axis=2).all(axis=0)]

    return left_frame.astype(np.uint8), right_frame.astype(np.uint8)