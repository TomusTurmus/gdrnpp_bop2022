import os.path as osp

import torch
from detectron2.config import LazyCall as L
from detectron2.solver.build import get_default_optimizer_params

from .yolox_base import train, val, test, model, dataloader, optimizer, lr_config, DATASETS
from det.yolox.data import build_yolox_test_loader, ValTransform
from det.yolox.data.datasets import Base_DatasetFromList
from detectron2.data import get_detection_dataset_dicts
from det.yolox.evaluators import YOLOX_COCOEvaluator
from lib.torch_utils.solver.ranger import Ranger

train.update(
    output_dir=osp.abspath(__file__).replace("configs", "output", 1)[0:-3],
    exp_name=osp.split(osp.abspath(__file__))[1][0:-3],
)
train.amp.enabled = True

train.init_checkpoint = "pretrained_models/yolox/yolox_x.pth"

DATASETS.TRAIN = []
DATASETS.TEST = ["custom_data_1"]

dataloader.train.dataset.lst.names = DATASETS.TRAIN
dataloader.train.total_batch_size = 32

model.head.num_classes = 1

optimizer = L(Ranger)(
    params=L(get_default_optimizer_params)(
        weight_decay_norm=0.0,
        weight_decay_bias=0.0,
    ),
    lr=0.001,
    weight_decay=0,
)

DATA_CFG = dict(
    custom_data_1="configs/custom_data_1/custom_data_1.json",
)

test.test_dataset_names = DATASETS.TEST
test.conf_thr = 0.001

dataloader.test = [
    L(build_yolox_test_loader)(
        dataset=L(Base_DatasetFromList)(
            split="test",
            lst=L(get_detection_dataset_dicts)(names=test_dataset_name, filter_empty=False),
            img_size="${test.test_size}",
            preproc=L(ValTransform)(legacy=False),
        ),
        total_batch_size=1,
        num_workers=4,
        pin_memory=True,
    )
    for test_dataset_name in test.test_dataset_names
]

dataloader.evaluator = [
    L(YOLOX_COCOEvaluator)(
        dataset_name=test_dataset_name,
        filter_scene=False,
    )
    for test_dataset_name in test.test_dataset_names
]