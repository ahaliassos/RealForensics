from collections import defaultdict

from torchmetrics.functional.classification.auroc import _auroc_compute, _auroc_update
import torch
import torch.distributed as dist
from torchmetrics import Metric


FF = ("Deepfakes", "FaceSwap", "Face2Face", "NeuralTextures")


class VideoLevelAUROC(Metric):
    def __init__(
            self,
            ds_types,
            num_classes=None,
            pos_label=None,
            average='macro',
            max_fpr=None,
            compute_on_step=True,
            dist_sync_on_step=False,
            process_group=None,
            dist_sync_fn=None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.num_classes = num_classes
        self.pos_label = pos_label
        self.average = average
        self.max_fpr = max_fpr

        allowed_average = (None, 'macro', 'weighted')
        if self.average not in allowed_average:
            raise ValueError(
                f'Argument `average` expected to be one of the following: {allowed_average} but got {average}'
            )

        if self.max_fpr is not None:
            if not isinstance(max_fpr, float) and 0 < max_fpr <= 1:
                raise ValueError(f"`max_fpr` should be a float in range (0, 1], got: {max_fpr}")

        self.mode = None
        assert "Real" in ds_types
        self.ds_types = ds_types
        for ds_type in ds_types:
            self.add_state(f"{ds_type.lower()}_preds", default=[], dist_reduce_fx=None)
            self.add_state(f"{ds_type.lower()}_targets", default=[], dist_reduce_fx=None)
            self.add_state(f"{ds_type.lower()}_indexes", default=[], dist_reduce_fx=None)

    def _ensemble_preds(self, preds, targets, indexes):
        preds = torch.cat(preds)
        targets = torch.cat(targets)
        indexes = torch.cat(indexes)

        preds_per_vid = defaultdict(list)
        targets_per_vid = {}
        for pred, target, index in zip(preds, targets, indexes):
            preds_per_vid[index.item()].append(pred)
            targets_per_vid[index.item()] = target
        # be very careful to have preds.shape == targets.shape; otherwise you get error about bool device
        preds = torch.stack(
            [torch.mean(torch.stack(pred_list), 0, keepdim=False) for pred_list in preds_per_vid.values()]).squeeze(1)
        targets = torch.stack([label for label in targets_per_vid.values()])
        return preds, targets

    def update(self, preds, targets, video_idxs, ds_type):
        """
        Update state with predictions and targets.
        Args:
            preds: Predictions from model (probabilities, or labels)
            targets: Ground truth labels
            ds_type: Dataset type (Real, FaceSwap, Deepfakes, etc)
            video_idxs: Video indices from the dataset
        """

        # very hacky - method requires inputs 0-1 but we want to average logits
        _, targets, mode = _auroc_update(torch.sigmoid(preds), targets)

        getattr(self, f"{ds_type.lower()}_preds").append(preds)
        getattr(self, f"{ds_type.lower()}_targets").append(targets)
        getattr(self, f"{ds_type.lower()}_indexes").append(video_idxs)

        if self.mode is not None and self.mode != mode:
            raise ValueError(
                'The mode of data (binary, multi-label, multi-class) should be constant, but changed'
                f' between batches from {self.mode} to {mode}'
            )
        self.mode = mode

    def compute(self):
        """
        Computes AUROC based on inputs passed in to ``update`` previously.
        """

        real_preds, real_targets = self._ensemble_preds(getattr(self, "real_preds"), getattr(self, "real_targets"),
                                                        getattr(self, "real_indexes"))
        all_preds, all_targets = [real_preds], [real_targets]
        aurocs = {}
        for ds_type in self.ds_types:
            if ds_type != "Real":
                fake_preds, fake_targets = self._ensemble_preds(getattr(self, f"{ds_type.lower()}_preds"),
                                                                getattr(self, f"{ds_type.lower()}_targets"),
                                                                getattr(self, f"{ds_type.lower()}_indexes"))
                preds = torch.cat([real_preds, fake_preds])
                targets = torch.cat([real_targets, fake_targets])
                aurocs[ds_type + "_AUC"] = _auroc_compute(
                    torch.sigmoid(preds), targets, self.mode, self.num_classes, self.pos_label, self.average,
                    self.max_fpr
                )
                if ds_type in FF:
                    all_preds.append(fake_preds)
                    all_targets.append(fake_targets)

        all_preds, all_targets = torch.cat(all_preds), torch.cat(all_targets)
        aurocs["Aggregate_AUC"] = _auroc_compute(
            torch.sigmoid(all_preds), all_targets, self.mode, self.num_classes, self.pos_label, self.average,
            self.max_fpr
        )
        return aurocs


class VideoLevelAUROCCDF:
    def __init__(
            self,
            ds_types,
            multi_gpu=False,
            num_classes=None,
            pos_label=None,
            average='macro',
            max_fpr=None,
    ):
        super().__init__()

        self.multi_gpu = multi_gpu

        self.num_classes = num_classes
        self.pos_label = pos_label
        self.average = average
        self.max_fpr = max_fpr

        allowed_average = (None, 'macro', 'weighted')
        if self.average not in allowed_average:
            raise ValueError(
                f'Argument `average` expected to be one of the following: {allowed_average} but got {average}'
            )

        if self.max_fpr is not None:
            if not isinstance(max_fpr, float) and 0 < max_fpr <= 1:
                raise ValueError(f"`max_fpr` should be a float in range (0, 1], got: {max_fpr}")

        self.mode = None
        assert "Real" in ds_types
        self.ds_types = ds_types
        for ds_type in ds_types:
            setattr(self, f"{ds_type.lower()}_preds", [])
            setattr(self, f"{ds_type.lower()}_targets", [])
            setattr(self, f"{ds_type.lower()}_indexes", [])

    def _ensemble_preds(self, preds, targets, indexes):
        preds, targets, indexes = torch.cat(preds), torch.cat(targets), torch.cat(indexes)

        if self.multi_gpu:
            preds, targets, indexes = concat_all_gather_var_len(preds), concat_all_gather_var_len(
                targets), concat_all_gather_var_len(indexes)

        preds_per_vid = defaultdict(list)
        targets_per_vid = {}
        for pred, target, index in zip(preds, targets, indexes):
            preds_per_vid[index.item()].append(pred)
            targets_per_vid[index.item()] = target
        # be very careful to have preds.shape == targets.shape; otherwise you get error about bool device
        preds = torch.stack(
            [torch.mean(torch.stack(pred_list), 0, keepdim=False) for pred_list in preds_per_vid.values()]).squeeze(1)
        targets = torch.stack([label for label in targets_per_vid.values()])
        return preds, targets

    def update(self, preds, targets, video_idxs, ds_type):
        """
        Update state with predictions and targets.
        Args:
            preds: Predictions from model (probabilities, or labels)
            targets: Ground truth labels
            ds_type: Dataset type (Real, FaceSwap, Deepfakes, etc)
            video_idxs: Video indices from the dataset
        """

        # very hacky - method requires inputs 0-1 but we want to average logits
        _, targets, mode = _auroc_update(torch.sigmoid(preds), targets)

        getattr(self, f"{ds_type.lower()}_preds").append(preds)
        getattr(self, f"{ds_type.lower()}_targets").append(targets)
        getattr(self, f"{ds_type.lower()}_indexes").append(video_idxs)

        if self.mode is not None and self.mode != mode:
            raise ValueError(
                'The mode of data (binary, multi-label, multi-class) should be constant, but changed'
                f' between batches from {self.mode} to {mode}'
            )
        self.mode = mode

    def compute(self):
        """
        Computes AUROC based on inputs passed in to ``update`` previously.
        """

        real_preds, real_targets = self._ensemble_preds(getattr(self, "real_preds"), getattr(self, "real_targets"),
                                                        getattr(self, "real_indexes"))
        aurocs = {}
        for ds_type in self.ds_types:
            if ds_type != "Real":
                fake_preds, fake_targets = self._ensemble_preds(getattr(self, f"{ds_type.lower()}_preds"),
                                                                getattr(self, f"{ds_type.lower()}_targets"),
                                                                getattr(self, f"{ds_type.lower()}_indexes"))
                preds = torch.cat([real_preds, fake_preds])
                targets = torch.cat([real_targets, fake_targets])
                aurocs[ds_type + "_AUC"] = _auroc_compute(
                    torch.sigmoid(preds), targets, self.mode, self.num_classes, self.pos_label, self.average,
                    self.max_fpr
                )
        return aurocs

    def reset(self):
        for ds_type in self.ds_types:
            setattr(self, f"{ds_type.lower()}_preds", [])
            setattr(self, f"{ds_type.lower()}_targets", [])
            setattr(self, f"{ds_type.lower()}_indexes", [])


class VideoLevelAUROCDFDC:
    def __init__(
            self,
            multi_gpu=False,
            num_classes=None,
            pos_label=None,
            average='macro',
            max_fpr=None,
    ):
        super().__init__()

        self.multi_gpu = multi_gpu

        self.num_classes = num_classes
        self.pos_label = pos_label
        self.average = average
        self.max_fpr = max_fpr

        allowed_average = (None, 'macro', 'weighted')
        if self.average not in allowed_average:
            raise ValueError(
                f'Argument `average` expected to be one of the following: {allowed_average} but got {average}'
            )

        if self.max_fpr is not None:
            if not isinstance(max_fpr, float) and 0 < max_fpr <= 1:
                raise ValueError(f"`max_fpr` should be a float in range (0, 1], got: {max_fpr}")

        self.mode = None

        self.preds = []
        self.targets = []
        self.indexes = []

    def _ensemble_preds(self, preds, targets, indexes):
        preds, targets, indexes = torch.cat(preds), torch.cat(targets), torch.cat(indexes)

        if self.multi_gpu:
            preds, targets, indexes = concat_all_gather_var_len(preds), concat_all_gather_var_len(
                targets), concat_all_gather_var_len(indexes)

        preds_per_vid = defaultdict(list)
        targets_per_vid = {}
        for pred, target, index in zip(preds, targets, indexes):
            preds_per_vid[index.item()].append(pred)
            targets_per_vid[index.item()] = target
        # be very careful to have preds.shape == targets.shape; otherwise you get error about bool device
        preds = torch.stack(
            [torch.mean(torch.stack(pred_list), 0, keepdim=False) for pred_list in preds_per_vid.values()]).squeeze(1)
        targets = torch.stack([label for label in targets_per_vid.values()])
        return preds, targets

    def update(self, preds, targets, video_idxs):
        """
        Update state with predictions and targets.
        Args:
            preds: Predictions from model (probabilities, or labels)
            targets: Ground truth labels
            video_idxs: Video indices from the dataset
        """

        # very hacky - method requires inputs 0-1 but we want to average logits
        _, targets, mode = _auroc_update(torch.sigmoid(preds), targets)

        self.preds.append(preds)
        self.targets.append(targets)
        self.indexes.append(video_idxs)

        if self.mode is not None and self.mode != mode:
            raise ValueError(
                'The mode of data (binary, multi-label, multi-class) should be constant, but changed'
                f' between batches from {self.mode} to {mode}'
            )
        self.mode = mode

    def compute(self):
        """
        Computes AUROC based on inputs passed in to ``update`` previously.
        """

        preds, targets = self._ensemble_preds(self.preds, self.targets, self.indexes)
        return {
            "DFDC_AUC": _auroc_compute(
                torch.sigmoid(preds), targets, self.mode, self.num_classes, self.pos_label, self.average, self.max_fpr
            )
        }

    def reset(self):
        self.preds = []
        self.targets = []
        self.indexes = []


class VideoLevelAcc(Metric):
    def __init__(
            self,
            ds_types,
            compute_on_step=True,
            dist_sync_on_step=False,
            process_group=None,
            dist_sync_fn=None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.ds_types = ds_types
        for ds_type in ds_types:
            self.add_state(f"{ds_type.lower()}_preds", default=[], dist_reduce_fx=None)
            self.add_state(f"{ds_type.lower()}_targets", default=[], dist_reduce_fx=None)
            self.add_state(f"{ds_type.lower()}_indexes", default=[], dist_reduce_fx=None)

    def _ensemble_preds(self, preds, targets, indexes):
        preds = torch.cat(preds)
        targets = torch.cat(targets)
        indexes = torch.cat(indexes)

        preds_per_vid = defaultdict(list)
        targets_per_vid = {}
        for pred, target, index in zip(preds, targets, indexes):
            preds_per_vid[index.item()].append(pred)
            targets_per_vid[index.item()] = target

        # be very careful to have preds.shape == targets.shape; otherwise you get error about bool device
        preds = torch.stack(
            [torch.mean(torch.stack(pred_list), 0, keepdim=False) for pred_list in preds_per_vid.values()]
        ).squeeze(1)
        targets = torch.stack([label for label in targets_per_vid.values()])
        return preds, targets

    def update(self, preds, targets, video_idxs, ds_type):
        """
        Update state with predictions and targets.
        Args:
            preds: Predictions from model (probabilities, or labels)
            targets: Ground truth labels
            ds_type: Dataset type (Real, FaceSwap, Deepfakes, etc)
            video_idxs: Video indices from the dataset
        """

        getattr(self, f"{ds_type.lower()}_preds").append(preds)
        getattr(self, f"{ds_type.lower()}_targets").append(targets)
        getattr(self, f"{ds_type.lower()}_indexes").append(video_idxs)

    def compute(self):
        """
        Computes AUROC based on inputs passed in to ``update`` previously.
        """

        real_preds, real_targets = self._ensemble_preds(
            getattr(self, "real_preds"), getattr(self, "real_targets"), getattr(self, "real_indexes")
        )

        all_preds, all_targets = [real_preds], [real_targets]

        accs = {}
        for ds_type in self.ds_types:
            if ds_type != "Real":
                fake_preds, fake_targets = self._ensemble_preds(
                    getattr(self, f"{ds_type.lower()}_preds"),
                    getattr(self, f"{ds_type.lower()}_targets"),
                    getattr(self, f"{ds_type.lower()}_indexes")
                )
                preds = torch.cat([real_preds, fake_preds])
                targets = torch.cat([real_targets, fake_targets])

                if ds_type in FF:
                    all_preds.append(fake_preds)
                    all_targets.append(fake_targets)

                accs[ds_type + "_Acc_Inclreal"] = ((preds > 0) == targets).float().mean()

        all_preds, all_targets = torch.cat(all_preds), torch.cat(all_targets)
        accs["Aggregate_Acc"] = ((all_preds > 0) == all_targets).float().mean()
        return accs


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    tensors_gather = torch.cat(tensors_gather, dim=0)
    return tensors_gather


@torch.no_grad()
def concat_all_gather_var_len(tensor):
    """
    Performs all_gather operation on tensors with variable clips lengths.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    # obtain Tensor number of frames of each rank
    world_size = dist.get_world_size()
    local_size = torch.tensor([tensor.size(0)], device=tensor.device)
    size_list = [torch.ones_like(local_size) for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensors_gather = [
        torch.ones(
            size=(max_size, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device
        ) for _ in range(world_size)
    ]
    if local_size != max_size:
        padding = torch.zeros(size=(max_size - local_size, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensors_gather, tensor)
    tensors_gather = [t[:local_size] for t, local_size in zip(tensors_gather, size_list)]
    tensors_gather = torch.cat(tensors_gather, dim=0)
    return tensors_gather
