_base_ = ["../../../_base_/gdrn_base.py"]

OUTPUT_DIR = "output/gdrn/custom_data_1/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_custom_data_1"

DATASETS = dict(
    TRAIN=(),
    TRAIN2=(),
    TRAIN2_RATIO=0.0,
    TEST=("custom_data_1",),
    SYM_OBJS=[],
)

DATA_CFG = dict(
    custom_data_1="configs/custom_data_1/custom_data_1.json",
)

INPUT = dict(
    WITH_DEPTH=True,
    CHANGE_BG_PROB=0.0,
    COLOR_AUG_PROB=0.0,
    RANDOM_FLIP="none",
)

MODEL = dict(
    LOAD_DETS_TEST=False,
    BBOX_TYPE="AMODAL_CLIP",
    POSE_NET=dict(
        NAME="GDRN_double_mask",
        XYZ_ONLINE=False,
    ),
)

TEST = dict(
    EVAL_PERIOD=0,
    VIS=False,
    TEST_BBOX_TYPE="est",
    USE_DEPTH_REFINE=False,
    SAVE_RESULTS_ONLY=False,
)