_base_ = ["../../../_base_/gdrn_base.py"]

OUTPUT_DIR = "output/gdrn/kitchen/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_kitchen"

INPUT = dict(
    DZI_PAD_SCALE=1.5,
    TRUNCATE_FG=True,
    CHANGE_BG_PROB=0.5,
    COLOR_AUG_PROB=0.8,
    COLOR_AUG_TYPE="code",
    WITH_DEPTH=True,
    RANDOM_FLIP="horizontal",
)

SOLVER = dict(
    IMS_PER_BATCH=6,
    TOTAL_EPOCHS=160,
    LR_SCHEDULER_NAME="flat_and_anneal",
    ANNEAL_METHOD="cosine",
    ANNEAL_POINT=0.72,
    OPTIMIZER_CFG=dict(_delete_=True, type="Ranger", lr=8e-4, weight_decay=0.01),
    WEIGHT_DECAY=0.0,
    WARMUP_FACTOR=0.001,
    WARMUP_ITERS=1000,
)

DATASETS = dict(
    TRAIN=("kitchen_train",),
    TRAIN2=(),
    TRAIN2_RATIO=0.0,
    TEST=("kitchen_val",),
    SYM_OBJS=[],
)

DATA_CFG = dict(
    kitchen_train="configs/gdrn/kitchen/kitchen_train.json",
    kitchen_val="configs/gdrn/kitchen/kitchen_val.json",
    kitchen_test="configs/gdrn/kitchen/kitchen_test.json",
)

MODEL = dict(
    LOAD_DETS_TEST=True,
    PIXEL_MEAN=[0.0, 0.0, 0.0],
    PIXEL_STD=[255.0, 255.0, 255.0],
    BBOX_TYPE="AMODAL_CLIP",
    POSE_NET=dict(
        NAME="GDRN_double_mask",
        XYZ_ONLINE=True,
        BACKBONE=dict(
            FREEZE=False,
            PRETRAINED="timm",
            INIT_CFG=dict(
                type="timm/convnext_base",
                pretrained=True,
                in_chans=3,
                features_only=True,
                out_indices=(3,),
            ),
        ),
        GEO_HEAD=dict(
            FREEZE=False,
            INIT_CFG=dict(
                type="TopDownDoubleMaskXyzRegionHead",
                in_dim=1024,
            ),
            NUM_REGIONS=64,
        ),
        PNP_NET=dict(
            INIT_CFG=dict(norm="GN", act="gelu"),
            REGION_ATTENTION=True,
            WITH_2D_COORD=True,
            ROT_TYPE="allo_rot6d",
            TRANS_TYPE="centroid_z",
        ),
        LOSS_CFG=dict(
            XYZ_LOSS_TYPE="L1",
            XYZ_LOSS_MASK_GT="visib",
            XYZ_LW=1.0,
            MASK_LOSS_TYPE="L1",
            MASK_LOSS_GT="trunc",
            MASK_LW=1.0,
            FULL_MASK_LOSS_TYPE="L1",
            FULL_MASK_LW=1.0,
            REGION_LOSS_TYPE="CE",
            REGION_LOSS_MASK_GT="visib",
            REGION_LW=1.0,
            PM_LOSS_SYM=False,
            PM_R_ONLY=True,
            PM_LW=1.0,
            CENTROID_LOSS_TYPE="L1",
            CENTROID_LW=1.0,
            Z_LOSS_TYPE="L1",
            Z_LW=1.0,
        ),
    ),
)

VAL = dict(
    DATASET_NAME="kitchen",
    SPLIT="val",
    SPLIT_TYPE="",
    SCRIPT_PATH="lib/pysixd/scripts/eval_pose_results_more.py",
    RESULTS_PATH="",
    TARGETS_FILENAME="manifests/kitchen_targets.json",
    ERROR_TYPES="ad,rete,proj",
    RENDERER_TYPE="cpp",
    N_TOP=1,
    EVAL_CACHED=False,
    SCORE_ONLY=False,
    EVAL_PRINT_ONLY=False,
    EVAL_PRECISION=False,
    USE_BOP=False,
    SAVE_BOP_CSV_ONLY=False,
)

TEST = dict(EVAL_PERIOD=0, VIS=False, TEST_BBOX_TYPE="est")
