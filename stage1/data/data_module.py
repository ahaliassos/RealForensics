from copy import deepcopy
import math
import os

from hydra.utils import instantiate
from omegaconf import OmegaConf
from pytorch_lightning import LightningDataModule
from pytorchvideo.data import make_clip_sampler
from pytorchvideo.data.clip_sampling import ClipInfo, ClipSampler
from pytorchvideo.transforms.functional import uniform_temporal_subsample
import torch
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from torchaudio.functional import amplitude_to_DB
from torchaudio.transforms import MelSpectrogram
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Grayscale,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    RandomGrayscale,
    RandomErasing,
    Resize,
)

from stage1.data.pytorchvideo_utils import labeled_video_dataset_with_len
from stage1.data.transforms import TimeMask, TimeMaskAudio, TimeMaskV2, TimeMaskAudioV2, FrequencyMask, FrequencyMaskV2, ZeroPadTemp


class ApplyTransformToKeyAug:
    def __init__(
            self,
            transform_video,
            transform_audio,
            transform_video_aug=None,
            transform_audio_aug=None,
            time_mask_video=None,
            time_mask_audio=None,
            args=None,
    ):
        self._transform_video = transform_video
        self._transform_audio = transform_audio
        self._transform_video_aug = transform_video_aug
        self._transform_audio_aug = transform_audio_aug
        self._time_mask_video = time_mask_video
        self._time_mask_audio = time_mask_audio
        self._pad_video = ZeroPadTemp(args.num_frames)
        self._pad_audio = ZeroPadTemp(args.num_frames * args.audio2video)
        self._args = args

    def __call__(self, x):
        original_video_size = x["video"].size(1)

        def subsample_audio(audio):
            return uniform_temporal_subsample(audio, original_video_size * self._args.audio2video, temporal_dim=1)

        x["video_aug"], _ = self._pad_video(self._transform_video_aug(x["video"]))
        x["video"], x["mask"] = self._pad_video(self._transform_video(x["video"]))

        x["audio_aug"], _ = self._pad_audio(subsample_audio(self._transform_audio_aug(x["audio"])))
        x["audio"], _ = self._pad_audio(subsample_audio(self._transform_audio(x["audio"])))

        if self._time_mask_video is not None:
            assert self._time_mask_audio is not None
            x["video_aug"], time_mask = self._time_mask_video(x["video_aug"])
            x["audio_aug"], _ = self._time_mask_audio(x["audio_aug"], time_mask)
            if self._args.time_mask_targets:
                x["video"], time_mask = self._time_mask_video(x["video"], time_mask)
                x["audio"], _ = self._time_mask_audio(x["audio"], time_mask)
                bool_mask = torch.ones(self._args.num_frames, dtype=x["mask"].dtype, device=x["mask"].device)
                bool_mask[time_mask] = 0
                x["mask"] *= bool_mask

        return x


class ApplyTransformToKey:
    def __init__(self, key, transform, args):
        self._key = key
        self._transform = transform
        self._args = args

    def __call__(self, x):
        x[self._key] = self._transform(x[self._key])
        if self._key == "audio":
            x[self._key] = uniform_temporal_subsample(
                x[self._key], self._args.num_frames * self._args.audio2video, temporal_dim=1
            )
        return x


class CenterClipSampler(ClipSampler):
    """
    Randomly samples clip of size clip_duration from the videos.
    """

    def __init__(self, clip_duration):
        super().__init__(clip_duration)

    def __call__(self, last_clip_time, video_duration, annotation):
        """
        Args:
            last_clip_time (float): Not used for CenterClipSampler.
            video_duration: (float): the duration (in seconds) for the video that's
                being sampled
            annotation (Dict): Not used by this sampler.
        Returns:
            clip_info (ClipInfo): includes the clip information of (clip_start_time,
            clip_end_time, clip_index, aug_index, is_last_clip). The times are in seconds.
            clip_index, aux_index and is_last_clip are always 0, 0 and True, respectively.
        """
        # clip_start_sec = (video_duration - self._clip_duration) / 2

        # return ClipInfo(clip_start_sec, clip_start_sec + self._clip_duration, 0, 0, True)
        return ClipInfo(0.08, 1.07, 0, 0, True)  # TODO: hardcoded because pytorchvideo makes off-by-one error


class DataModule(LightningDataModule):

    def __init__(self, cfg, root):
        super().__init__()
        self.cfg = cfg
        self.root = root

    def _make_transforms(self, mode):
        args = self.cfg.data
        video_transform, video_transform_aug = self._video_transform(mode)
        audio_transform, audio_transform_aug = self._audio_transform()

        time_mask_video = time_mask_audio = None
        if (args.mask_version == "v1" and args.n_time_mask_video != 0) or (
                args.mask_version == "v2" and not math.isclose(args.time_mask_prob_video, 0.0)):
            if args.mask_version == "v1":
                time_mask_video = TimeMask(
                    T=args.time_mask_video, n_mask=args.n_time_mask_video, replace_with_zero=True
                )
            else:
                time_mask_video = TimeMaskV2(
                    p=args.time_mask_prob_video, T=args.time_mask_video, replace_with_zero=True
                )

        if (args.mask_version == "v1" and args.n_time_mask_audio != 0) or (
                args.mask_version == "v2" and not math.isclose(args.time_mask_prob_audio, 0.0)):
            if args.mask_version == "v1":
                time_mask_audio = TimeMaskAudio(
                    T=args.time_mask_audio,
                    n_mask=args.n_time_mask_audio,
                    downsample=args.audio2video,
                    replace_with_zero=True
                )
            else:
                time_mask_audio = TimeMaskAudioV2(
                    p=args.time_mask_prob_audio,
                    T=args.time_mask_audio,
                    downsample=args.audio2video,
                    replace_with_zero=True
                )

        return ApplyTransformToKeyAug(
            video_transform,
            audio_transform,
            video_transform_aug,
            audio_transform_aug,
            time_mask_video,
            time_mask_audio,
            args,
        )

    def _make_transforms_lrw(self, mode):
        args = self.cfg.data
        transform_video, transform_video_aug = self._video_transform(mode)
        transform_audio, transform_audio_aug = self._audio_transform()
        if mode == "train":
            transform_video = transform_video_aug
            transform_audio = transform_audio_aug

        return Compose([
            ApplyTransformToKey("video", transform_video, args), ApplyTransformToKey("audio", transform_audio, args)
        ])

    def _video_transform(self, mode):
        args = self.cfg.data
        transform = [
            # UniformTemporalSubsample(args.num_frames),
            Lambda(lambda x: x / 255.),
        ] + (
            [
                RandomCrop(args.crop_type.random_crop_dim),
                Resize(args.crop_type.resize_dim),
                RandomHorizontalFlip(args.horizontal_flip_prob)
            ]
            if mode == "train" else [CenterCrop(args.crop_type.random_crop_dim), Resize(args.crop_type.resize_dim)]
        )
        if self.cfg.data.channel.in_video_channels == 1:
            transform.extend([Lambda(lambda x: x.transpose(0, 1)), Grayscale(), Lambda(lambda x: x.transpose(0, 1))])

        if args.channel.in_video_channels != 1 and math.isclose(args.channel.grayscale_prob, 1.0):
            transform.extend(
                [
                    Lambda(lambda x: x.transpose(0, 1)),
                    RandomGrayscale(args.channel.grayscale_prob),
                    Lambda(lambda x: x.transpose(0, 1))
                ]
            )

        transform_aug = deepcopy(transform)
        if args.channel.in_video_channels != 1 and not math.isclose(args.channel.grayscale_prob, 0.0):
            transform_aug.extend(
                [
                    Lambda(lambda x: x.transpose(0, 1)),
                    RandomGrayscale(args.channel.grayscale_prob),
                    Lambda(lambda x: x.transpose(0, 1))
                ]
            )
        transform_aug.append(instantiate(args.channel.obj))
        if not math.isclose(args.crop_type.random_erasing_prob, 0.0):
            transform_aug.append(
                RandomErasing(
                    p=args.crop_type.random_erasing_prob,
                    scale=OmegaConf.to_object(args.crop_type.random_erasing_scale)
                )
            )

        transform.append(instantiate(args.channel.obj))

        return Compose(transform), Compose(transform_aug)

    def _audio_transform(self):
        args = self.cfg.data
        transform = [
            MelSpectrogram(
                sample_rate=args.dataset.sample_rate,
                win_length=args.win_length,
                hop_length=int(round(args.dataset.sample_rate / args.dataset.fps / args.audio2video)),
                n_fft=args.n_fft,
                n_mels=args.n_mels,
            ),
            Lambda(
                lambda x: x[:, :-(x.size(1) - args.dataset.fps * args.audio2video)]
                if x.size(1) > args.dataset.fps * args.audio2video else x
            ),
            Lambda(lambda x: amplitude_to_DB(
                x, multiplier=10, amin=1e-10, db_multiplier=math.log10(max(1e-10, torch.max(x).item())), top_db=80
            )),
            Lambda(lambda x: (x + 40) / 40),
            Lambda(lambda x: x.transpose(1, 0).unsqueeze(0)),  # (F, T) -> (1, T, F)
        ]

        transform_aug = deepcopy(transform)
        if (args.mask_version == "v1" and args.n_freq_mask != 0) or (args.mask_version == "v2" and not math.isclose(args.freq_mask_prob_audio, 0.0)):
            if args.mask_version == "v1":
                transform_aug.append(FrequencyMask(F=args.freq_mask, n_mask=args.n_freq_mask, replace_with_zero=True))
            else:
                transform_aug.append(
                    FrequencyMaskV2(p=args.freq_mask_prob_audio, F=args.freq_mask, replace_with_zero=True)
                )

        return Compose(transform), Compose(transform_aug)

    def _dataloader(self, ds, batch_size, drop_last=False):
        return DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=drop_last,
        )

    def train_dataloader(self):
        args = self.cfg.data.dataset
        ds_root = os.path.join(args.root.root, args.name)
        sampler = DistributedSampler if self.cfg.gpus > 1 else RandomSampler

        offset = 0.04  # needed because pytorchvideo makes off-by-one error
        self.train_ds = labeled_video_dataset_with_len(
            data_path=os.path.join(self.root, ds_root, self.cfg.data.crop_type.train_csv),
            clip_sampler=make_clip_sampler("random", self.cfg.data.num_frames / args.fps - offset),  # hacky
            video_path_prefix=os.path.join(self.root, ds_root, self.cfg.data.crop_type.video_dir),
            transform=self._make_transforms(mode="train"),
            video_sampler=sampler,
        )
        return self._dataloader(self.train_ds, self.cfg.batch_size // self.cfg.gpus)

    def val_dataloader(self):
        args = self.cfg.data.dataset
        lrw_root = os.path.join(args.root.root, "LRW500")
        sampler = DistributedSampler if self.cfg.gpus else RandomSampler
        self.ds_val = labeled_video_dataset_with_len(
            data_path=os.path.join(self.root, lrw_root, self.cfg.data.crop_type.val_csv),
            clip_sampler=CenterClipSampler(self.cfg.data.num_frames / args.fps),
            video_path_prefix=os.path.join(self.root, lrw_root, self.cfg.data.crop_type.video_dir),
            transform=self._make_transforms_lrw(mode="val"),
            video_sampler=sampler,
        )
        return self._dataloader(self.ds_val, self.cfg.batch_size // self.cfg.gpus)

    def test_dataloader(self):
        args = self.cfg.data.dataset
        lrw_root = os.path.join(args.root.root, "LRW500")
        sampler = DistributedSampler if self.cfg.gpus else RandomSampler
        self.ds_test = labeled_video_dataset_with_len(
            data_path=os.path.join(self.root, lrw_root, self.cfg.data.crop_type.val_csv),
            clip_sampler=CenterClipSampler(self.cfg.data.num_frames / args.fps),
            video_path_prefix=os.path.join(self.root, lrw_root, self.cfg.data.crop_type.video_dir),
            transform=self._make_transforms_lrw(mode="val"),
            video_sampler=sampler,
        )
        return self._dataloader(self.ds_test, self.cfg.batch_size // self.cfg.gpus)
