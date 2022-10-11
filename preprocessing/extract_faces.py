import argparse
import os

import cv2
from pathlib import Path
import numpy as np
from tqdm import tqdm


SAMPLE_RATE = 16_000


def load_args():
    parser = argparse.ArgumentParser(description='Pre-processing')
    parser.add_argument('--root-dir', default=None, help='video directory')
    parser.add_argument('--mean-face', default='./preprocessing/20words_mean_face.npy', help='mean face path')
    parser.add_argument('--crop-width', default=150, type=int, help='width of face crop')
    parser.add_argument('--crop-height', default=150, type=int, help='height of face crop')
    parser.add_argument('--start-idx', default=15, type=int, help='start of landmark index')
    parser.add_argument('--stop-idx', default=68, type=int, help='end of landmark index')
    parser.add_argument('--window-margin', default=12, type=int, help='window margin for smoothed landmarks')

    args = parser.parse_args()
    return args


def save_video_lossless(filename, vid, frames_per_second, audio_path=None):
    fourcc = cv2.VideoWriter_fourcc("F", "F", "V", "1")
    writer = cv2.VideoWriter(filename + ".avi", fourcc, frames_per_second, (vid[0].shape[1], vid[0].shape[0]))
    for frame in vid:
        writer.write(frame)
    writer.release()  # close the writer

    if audio_path:
        cmd = f'ffmpeg -y -loglevel warning -i {filename + ".avi"} -i {audio_path} -c copy {filename + "temp.avi"}'
        os.system(cmd)
        os.remove(filename + ".avi")
        os.rename(filename + "temp.avi", filename + ".avi")


def affine_transform(frame, landmarks, reference, grayscale=False, target_size=(256, 256),
                     reference_size=(256, 256), stable_points=(28, 33, 36, 39, 42, 45, 48, 54),
                     interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT,
                     border_value=0):
    # Prepare everything
    if grayscale and frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    stable_reference = np.vstack([reference[x] for x in stable_points])
    stable_reference[:, 0] -= (reference_size[0] - target_size[0]) / 2.0
    stable_reference[:, 1] -= (reference_size[1] - target_size[1]) / 2.0

    # Warp the face patch and the landmarks
    transform = cv2.estimateAffinePartial2D(np.vstack([landmarks[x] for x in stable_points]),
                                            stable_reference, method=cv2.LMEDS)[0]
    transformed_frame = cv2.warpAffine(frame, transform, dsize=(target_size[0], target_size[1]),
                                flags=interpolation, borderMode=border_mode, borderValue=border_value)
    transformed_landmarks = np.matmul(landmarks, transform[:, :2].transpose()) + transform[:, 2].transpose()

    return transformed_frame, transformed_landmarks


def cut_patch(img, landmarks, height, width, threshold=5):
    center_x, center_y = np.mean(landmarks, axis=0)

    if center_y - height < 0:
        center_y = height
    if center_y - height < 0 - threshold:
        raise Exception('too much bias in height')
    if center_x - width < 0:
        center_x = width
    if center_x - width < 0 - threshold:
        raise Exception('too much bias in width')

    if center_y + height > img.shape[0]:
        center_y = img.shape[0] - height
    if center_y + height > img.shape[0] + threshold:
        raise Exception('too much bias in height')
    if center_x + width > img.shape[1]:
        center_x = img.shape[1] - width
    if center_x + width > img.shape[1] + threshold:
        raise Exception('too much bias in width')

    cutted_img = np.copy(img[int(round(center_y) - round(height)): int(round(center_y) + round(height)),
                         int(round(center_x) - round(width)): int(round(center_x) + round(width))])
    return cutted_img


def crop_patch(frames, landmarks, reference, args):
    sequence = []
    length = min(len(landmarks), len(frames))
    for frame_idx in range(length):
        frame = frames[frame_idx]
        window_margin = min(args.window_margin // 2, frame_idx, len(landmarks) - 1 - frame_idx)
        smoothed_landmarks = np.mean(
            [landmarks[x] for x in range(frame_idx - window_margin, frame_idx + window_margin + 1)], axis=0
        )
        smoothed_landmarks += landmarks[frame_idx].mean(axis=0) - smoothed_landmarks.mean(axis=0)
        transformed_frame, transformed_landmarks = affine_transform(
            frame, smoothed_landmarks, reference, grayscale=False
        )
        sequence.append(
            cut_patch(
                transformed_frame,
                transformed_landmarks[args.start_idx:args.stop_idx],
                args.crop_height//2,
                args.crop_width//2,
            )
        )
    return np.array(sequence)


def get_video_clip(video_filename):
    cap = cv2.VideoCapture(video_filename)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame.copy())
    cap.release()
    return frames


def main():
    args = load_args()
    reference = np.load(args.mean_face)

    video_root = os.path.join(args.root_dir, "videos")
    landmarks_root = os.path.join(args.root_dir, "landmarks")
    target_root = os.path.join(args.root_dir, "cropped_faces")

    for path in tqdm(Path(video_root).rglob("*.mp4")):
        relpath = os.path.relpath(path, video_root)

        video_path = os.path.join(video_root, relpath)
        landmarks_path = os.path.join(landmarks_root, relpath[:-4] + ".npy")

        video = get_video_clip(video_path)

        landmarks = np.load(landmarks_path)
        sequence = crop_patch(video, landmarks, reference, args)

        target_dir = os.path.join(target_root, relpath[:-4])
        os.makedirs(os.path.dirname(target_dir), exist_ok=True)

        audio_path = None
        if args.root_dir.endswith("LRW/"):  # extract audio
            audio_path = video_path[:-4] + ".wav"
            cmd = f"ffmpeg -i {video_path} {audio_path}"
            os.system(cmd)

        save_video_lossless(target_dir, sequence, 25, audio_path)

        if args.root_dir.endswith("LRW/"):
            os.remove(audio_path)


if __name__ == "__main__":
    main()
