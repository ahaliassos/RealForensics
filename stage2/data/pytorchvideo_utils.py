# adapted from https://github.com/facebookresearch/pytorchvideo/tree/main/pytorchvideo/data

import io
import math
import pathlib

from iopath.common.file_io import g_pathmgr
import numpy as np
from pytorchvideo.data.frame_video import FrameVideo
from pytorchvideo.data.video import Video, VideoPathHandler
from pytorchvideo.data import labeled_video_dataset, LabeledVideoDataset
from pytorchvideo.data.encoded_video_pyav import EncodedVideoPyAV
from pytorchvideo.data.labeled_video_paths import LabeledVideoPaths
from pytorchvideo.data.utils import pts_to_secs, secs_to_pts, thwc_to_cthw
import torch


class LabeledVideoDatasetWithLen(LabeledVideoDataset):
    def __len__(self):
        return super().num_videos


class LabeledVideoPathsDF(LabeledVideoPaths):
    @classmethod
    def from_types(cls, types, compression, crop_type, real_path, fake_path=None, deeper_path=None):
        video_paths_and_label = []
        for typ in types:
            if typ.lower() == "real":
                file_path = real_path
            elif typ.lower() == "deeperforensics":
                file_path = deeper_path
            else:
                file_path = fake_path

            assert g_pathmgr.exists(file_path), f"{file_path} not found."

            with g_pathmgr.open(file_path, "r") as f:
                for path_label in f.read().splitlines():
                    line_split = path_label.rsplit(None, 1)

                    # The video path file may not contain labels (e.g. for a test split). We
                    # assume this is the case if only 1 path is found and set the label to
                    # -1 if so.
                    if len(line_split) == 1:
                        file_path = line_split[0]
                        label = -1
                    else:
                        file_path, label = line_split

                    video_paths_and_label.append(("/".join([typ, compression, crop_type, file_path]), int(label)))

            assert (
                    len(video_paths_and_label) > 0
            ), f"Failed to load dataset from {file_path}."
        return cls(video_paths_and_label)


def labeled_video_dataset_with_fix(
    data_path: str,
    clip_sampler,
    video_sampler=torch.utils.data.RandomSampler,
    transform=None,
    video_path_prefix="",
    decode_audio=True,
    decoder="pyav",
    with_length=True
):
    labeled_video_paths = LabeledVideoPaths.from_path(data_path)
    labeled_video_paths.path_prefix = video_path_prefix
    ds_cls = LabeledVideoDatasetWithLen if with_length else LabeledVideoDataset
    dataset = ds_cls(
        labeled_video_paths,
        clip_sampler,
        video_sampler,
        transform,
        decode_audio=decode_audio,
        decoder=decoder,
    )

    dataset.video_path_handler = VideoPathHandlerMarginFix()
    return dataset


def labeled_video_dataset_types(
    types,
    real_path: str,
    fake_path: str,
    deeper_path: str,
    clip_sampler,
    video_sampler=torch.utils.data.RandomSampler,
    transform=None,
    video_path_prefix="",
    decoder="pyav",
    compression="c23",
    crop_type="cropped_mouths",
    with_length=True,
):
    labeled_video_paths = LabeledVideoPathsDF.from_types(
        types, compression, crop_type, real_path, fake_path, deeper_path
    )

    labeled_video_paths.path_prefix = video_path_prefix
    ds_cls = LabeledVideoDatasetWithLen if with_length else LabeledVideoDataset
    dataset = ds_cls(
        labeled_video_paths,
        clip_sampler,
        video_sampler,
        transform,
        decode_audio=False,
        decoder=decoder,
    )
    dataset.video_path_handler = VideoPathHandlerMarginFix()
    return dataset


class EncodedVideoPyAVMarginFix(EncodedVideoPyAV):
    def _pyav_decode_video(self, start_secs=0.0, end_secs=math.inf):
        """
        Selectively decodes a video between start_pts and end_pts in time units of the
        self._video's timebase.
        """
        video_and_pts = None
        audio_and_pts = None
        try:
            pyav_video_frames, _ = _pyav_decode_stream(
                self._container,
                secs_to_pts(start_secs, self._video_time_base, self._video_start_pts),
                secs_to_pts(end_secs, self._video_time_base, self._video_start_pts),
                self._container.streams.video[0],
                {"video": 0},
            )
            if len(pyav_video_frames) > 0:
                video_and_pts = [
                    (torch.from_numpy(frame.to_rgb().to_ndarray()), frame.pts)
                    for frame in pyav_video_frames
                ]

            if self._has_audio:
                pyav_audio_frames, _ = _pyav_decode_stream(
                    self._container,
                    secs_to_pts(
                        start_secs, self._audio_time_base, self._audio_start_pts
                    ),
                    secs_to_pts(end_secs, self._audio_time_base, self._audio_start_pts),
                    self._container.streams.audio[0],
                    {"audio": 0},
                )

                if len(pyav_audio_frames) > 0:
                    audio_and_pts = [
                        (
                            torch.from_numpy(np.mean(frame.to_ndarray(), axis=0)),
                            frame.pts,
                        )
                        for frame in pyav_audio_frames
                    ]

        except Exception as e:
            raise e

        return video_and_pts, audio_and_pts


def _pyav_decode_stream(
        container,
        start_pts,
        end_pts,
        stream,
        stream_name,
        buffer_size=0,
):
    """
    Decode the video with PyAV decoder.
    Args:
        container (container): PyAV container.
        start_pts (int): the starting Presentation TimeStamp to fetch the
            video frames.
        end_pts (int): the ending Presentation TimeStamp of the decoded frames.
        stream (stream): PyAV stream.
        stream_name (dict): a dictionary of streams. For example, {"video": 0}
            means video stream at stream index 0.
    Returns:
        result (list): list of decoded frames.
        max_pts (int): max Presentation TimeStamp of the video sequence.
    """

    # Seeking in the stream is imprecise. Thus, seek to an earlier pts by a
    # margin pts.
    margin = 1024 if "audio" in stream_name else 1  # This is what's different from pytorchvideo
    seek_offset = max(start_pts - margin, 0)
    container.seek(int(seek_offset), any_frame=False, backward=True, stream=stream)
    frames = {}
    max_pts = 0
    for frame in container.decode(**stream_name):
        max_pts = max(max_pts, frame.pts)
        if frame.pts >= start_pts and frame.pts <= end_pts:
            frames[frame.pts] = frame
        elif frame.pts > end_pts:
            break

    result = [frames[pts] for pts in sorted(frames)]
    return result, max_pts


class EncodedVideoMarginFix(Video):
    """
    EncodedVideo is an abstraction for accessing clips from an encoded video.
    It supports selective decoding when header information is available.
    """

    @classmethod
    def from_path(
        cls, file_path: str, decode_audio: bool = True, decoder: str = "pyav"
    ):
        """
        Fetches the given video path using PathManager (allowing remote uris to be
        fetched) and constructs the EncodedVideo object.
        Args:
            file_path (str): a PathManager file-path.
        """
        # We read the file with PathManager so that we can read from remote uris.
        with g_pathmgr.open(file_path, "rb") as fh:
            video_file = io.BytesIO(fh.read())

        video_cls = EncodedVideoPyAVMarginFix
        return video_cls(video_file, pathlib.Path(file_path).name, decode_audio)


class VideoPathHandlerMarginFix:
    """
    Utility class that handles all deciphering and caching of video paths for
    encoded and frame videos.
    """
    def __init__(self) -> None:
        # Pathmanager isn't guaranteed to be in correct order,
        # sorting is expensive, so we cache paths in case of frame video and reuse.
        self.path_order_cache = {}

    def video_from_path(self, filepath, decode_audio=False, decoder="pyav", fps=30):
        return EncodedVideoMarginFix.from_path(filepath, decode_audio, decoder)