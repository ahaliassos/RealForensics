from copy import deepcopy
import math
import os

from hydra.utils import instantiate
from omegaconf import OmegaConf
from pytorch_lightning import LightningDataModule
from pytorchvideo.data import make_clip_sampler
from pytorchvideo.data.clip_sampling import ClipInfo, UniformClipSampler
from pytorchvideo.transforms import RemoveKey
import torch
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Grayscale,
    RandomCrop,
    RandomHorizontalFlip,
    RandomGrayscale,
    RandomErasing,
    Resize,
    RandomApply,
)

from stage2.data.pytorchvideo_utils import labeled_video_dataset_types, labeled_video_dataset_with_fix
from stage2.data.transforms import TimeMask, TimeMaskV2, ZeroPadTemp, LambdaModule

FF_TYPES = {"Deepfakes", "FaceSwap", "Face2Face", "NeuralTextures", "FaceShifter", "DeeperForensics"}


class ApplyTransformToKeyAug:
    def __init__(
            self,
            transform_video,
            transform_video_aug=None,
            time_mask_video=None,
            args=None,
    ):
        self._transform_video = transform_video
        self._transform_video_aug = transform_video_aug
        self._time_mask_video = time_mask_video
        self._pad_video = ZeroPadTemp(args.num_frames)
        self._args = args

    def __call__(self, x):
        del x["video_name"]
        x["video_aug"], _ = self._pad_video(self._transform_video_aug(x["video"]))
        x["video"], x["mask"] = self._pad_video(self._transform_video(x["video"]))

        if self._time_mask_video is not None and torch.rand(1) < self._args.time_mask_prob_video:
            x["video_aug"], time_mask = self._time_mask_video(x["video_aug"])
            bool_mask = torch.ones(self._args.num_frames, dtype=x["mask"].dtype, device=x["mask"].device)
            bool_mask[time_mask] = 0
            x["mask"] *= bool_mask

        return x


class UniformClipSamplerWithDuration(UniformClipSampler):
    """
    Evenly splits the video into clips of size clip_duration.
    """

    def __init__(
            self,
            clip_duration,
            video_duration,
            stride=None,
            backpad_last=False,
            eps=1e-6,
    ):
        super().__init__(clip_duration, stride, backpad_last, eps)
        self._video_duration = video_duration

    def __call__(self, last_clip_time, video_duration, annotation):
        video_duration = self._video_duration
        clip_start, clip_end = self._clip_start_end(
            last_clip_time, video_duration, backpad_last=self._backpad_last
        )

        # if they both end at the same time - it's the last clip
        _, next_clip_end = self._clip_start_end(
            clip_end, video_duration, backpad_last=self._backpad_last
        )
        if self._backpad_last:
            is_last_clip = abs(next_clip_end - clip_end) < self._eps
        else:
            is_last_clip = next_clip_end > video_duration

        clip_index = self._current_clip_index
        self._current_clip_index += 1

        return ClipInfo(clip_start, clip_end, clip_index, 0, is_last_clip)


class DataModule(LightningDataModule):
    def __init__(self, cfg, root):
        super().__init__()
        self.cfg = cfg
        self.root = root

    def _make_transforms(self, mode):
        args = self.cfg.data
        video_transform, video_transform_aug = self._video_transform(mode)

        time_mask_video = None
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

        transform = ApplyTransformToKeyAug(
            video_transform,
            video_transform_aug,
            time_mask_video,
            args,
        ), RemoveKey("audio")

        return Compose(transform)

    def _video_transform(self, mode):
        args = self.cfg.data
        transform = [
                        # UniformTemporalSubsample(args.num_frames),
                        LambdaModule(lambda x: x / 255.),
                    ] + (
                        [
                            RandomCrop(args.crop_type.random_crop_dim),
                            Resize(args.crop_type.resize_dim),
                            RandomHorizontalFlip(args.horizontal_flip_prob)
                        ]
                        if mode == "train" else [CenterCrop(args.crop_type.random_crop_dim),
                                                 Resize(args.crop_type.resize_dim)]
                    )
        if self.cfg.data.channel.in_video_channels == 1:
            transform.extend(
                [LambdaModule(lambda x: x.transpose(0, 1)), Grayscale(), LambdaModule(lambda x: x.transpose(0, 1))])

        if args.channel.in_video_channels != 1 and math.isclose(args.channel.grayscale_prob, 1.0):
            transform.extend(
                [
                    LambdaModule(lambda x: x.transpose(0, 1)),
                    RandomGrayscale(args.channel.grayscale_prob),
                    LambdaModule(lambda x: x.transpose(0, 1))
                ]
            )

        transform_aug = []

        if args.channel.in_video_channels != 1 and not math.isclose(args.channel.grayscale_prob, 0.0):
            transform_aug.extend(
                [
                    LambdaModule(lambda x: x.transpose(0, 1)),
                    RandomGrayscale(args.channel.grayscale_prob),
                    LambdaModule(lambda x: x.transpose(0, 1))
                ]
            )

        transform_aug = [
            *deepcopy(transform),
            RandomApply(torch.nn.ModuleList(transform_aug), p=args.aug_prob),
            instantiate(args.channel.obj)
        ]

        if not math.isclose(args.crop_type.random_erasing_prob, 0.0):
            transform_aug.append(
                RandomErasing(
                    p=args.crop_type.random_erasing_prob,
                    scale=OmegaConf.to_object(args.crop_type.random_erasing_scale)
                )
            )

        transform.append(instantiate(args.channel.obj))

        return Compose(transform), Compose(transform_aug)

    def _dataloader(self, ds, drop_last=False, sampler=None, scale=1, shuffle=False):
        return DataLoader(
            ds,
            batch_size=int((scale * self.cfg.batch_size) // self.cfg.gpus),
            num_workers=self.cfg.num_workers // self.cfg.gpus,
            pin_memory=True,
            drop_last=drop_last,
            sampler=sampler,
            shuffle=shuffle,
        )

    def train_dataloader(self):
        sampler = DistributedSampler if self.cfg.gpus > 1 else RandomSampler

        fake_types_train = self.cfg.data.dataset_df.fake_types_train
        if self.cfg.data.all_but:
            fake_types_train.remove(self.cfg.data.all_but)
        offset = 0.04

        loaders = []
        if not self.cfg.only_df:
            # ssl
            args = self.cfg.data.dataset
            ds_root = os.path.join(self.cfg.data.root.root, args.name)
            train_ds = labeled_video_dataset_with_fix(
                data_path=os.path.join(self.root, ds_root, self.cfg.data.crop_type.train_csv),
                clip_sampler=make_clip_sampler("random", self.cfg.data.num_frames / args.fps - offset),  # hacky
                video_path_prefix=os.path.join(self.root, ds_root, self.cfg.data.crop_type.video_dir),
                transform=self._make_transforms(mode="train"),
                video_sampler=sampler,
                with_length=False
            )

            loaders.append(self._dataloader(train_ds, drop_last=False, scale=self.cfg.model.relative_bs))

        # dataset_df
        ds_types = ["Real"] + OmegaConf.to_object(fake_types_train)

        dataset = labeled_video_dataset_types(
            types=ds_types,
            real_path=os.path.join(self.root, "data", "Forensics", 'csv_files', 'train_real.csv'),
            fake_path=os.path.join(self.root, "data", "Forensics", 'csv_files', 'train_fake.csv'),
            deeper_path=os.path.join(self.root, "data", "Forensics", 'csv_files', 'train_dfo.csv'),
            clip_sampler=make_clip_sampler("random", self.cfg.data.num_frames / self.cfg.data.dataset_df.fps - offset),
            video_path_prefix=os.path.join(self.root, "data", "Forensics"),
            transform=self._make_transforms(mode="train"),
            video_sampler=sampler,
            compression=self.cfg.data.dataset_df.ds_type,
            crop_type=self.cfg.data.crop_type.video_dir_df,
            with_length=False
        )

        loaders.append(self._dataloader(dataset, drop_last=False))

        return loaders

    def val_dataloader(self):
        sampler = DistributedSampler if self.cfg.gpus > 1 else RandomSampler

        offset = 0.04

        loaders = []
        # FF++
        for ds_type in self.cfg.data.dataset_df.types_val:
            if self.cfg.data.dataset_df.only_ff_val and ds_type in ("DeeperForensics", "FaceShifter"):
                continue
            if ds_type == "Real":
                csv_type = "real"
            elif ds_type == "DeeperForensics":
                csv_type = "dfo"
            else:
                csv_type = "fake"

            ds = labeled_video_dataset_with_fix(
                data_path=os.path.join(self.root, "data", "Forensics", "csv_files", f"val_{csv_type}.csv"),
                clip_sampler=UniformClipSamplerWithDuration(
                    self.cfg.data.num_frames / self.cfg.data.dataset_df.fps - offset,
                    110 / self.cfg.data.dataset_df.fps,
                    backpad_last=True,
                ),
                video_path_prefix=os.path.join(
                    self.root,
                    "data",
                    "Forensics",
                    ds_type,
                    self.cfg.data.dataset_df.ds_type,
                    self.cfg.data.crop_type.video_dir_df
                ),
                transform=self._make_transforms(mode="val"),
                video_sampler=sampler,
                with_length=False
            )
            loaders.append(self._dataloader(ds))

        return loaders

    def test_dataloader(self):
        sampler = DistributedSampler if self.cfg.gpus > 1 else RandomSampler

        offset = 0.04

        loaders = []
        # FF++
        for ds_type in self.cfg.data.dataset_df.types_val:
            if self.cfg.data.dataset_df.only_ff_val and ds_type in ("DeeperForensics", "FaceShifter"):
                continue
            if ds_type == "Real":
                csv_type = "real"
            elif ds_type == "DeeperForensics":
                csv_type = "dfo"
            else:
                csv_type = "fake"

            ds = labeled_video_dataset_with_fix(
                data_path=os.path.join(self.root, "data", "Forensics", "csv_files", f"test_{csv_type}.csv"),
                clip_sampler=UniformClipSamplerWithDuration(
                    self.cfg.data.num_frames / self.cfg.data.dataset_df.fps - offset,
                    110 / self.cfg.data.dataset_df.fps,
                    backpad_last=True,
                ),
                video_path_prefix=os.path.join(
                    self.root,
                    "data",
                    "Forensics",
                    ds_type,
                    self.cfg.data.dataset_df.ds_type,
                    self.cfg.data.crop_type.video_dir_df
                ),
                transform=self._make_transforms(mode="val"),
                video_sampler=sampler,
                with_length=False
            )
            loaders.append(self._dataloader(ds))

        if self.cfg.data.dataset_df.cdf_dfdc_test:
            # CelebDFv2
            for ds_type in ("Real", "Fake"):
                ds = labeled_video_dataset_with_fix(
                    data_path=os.path.join(self.root, "data", "CelebDF", "csv_files", f"test_{ds_type.lower()}.csv"),
                    clip_sampler=make_clip_sampler(
                        "uniform", self.cfg.data.num_frames / self.cfg.data.dataset_df.fps - offset, None, True
                    ),
                    video_path_prefix=os.path.join(
                        self.root,
                        "data",
                        "CelebDF",
                        ds_type,
                        self.cfg.data.crop_type.video_dir_df
                    ),
                    transform=self._make_transforms(mode="val"),
                    video_sampler=sampler,
                    with_length=False
                )
                loaders.append(self._dataloader(ds))

            # DFDC
            ds = labeled_video_dataset_with_fix(
                data_path=os.path.join(os.path.join(self.root, "data", "DFDC", "csv_files", "test.csv")),
                clip_sampler=make_clip_sampler(
                    "uniform", self.cfg.data.num_frames / self.cfg.data.dataset_df.fps - offset, None, True
                ),
                video_path_prefix=os.path.join(
                    self.root,
                    "data",
                    "DFDC",
                    self.cfg.data.crop_type.video_dir_df
                ),
                transform=self._make_transforms(mode="val"),
                video_sampler=sampler,
                with_length=False
            )
            loaders.append(self._dataloader(ds))

        return loaders
