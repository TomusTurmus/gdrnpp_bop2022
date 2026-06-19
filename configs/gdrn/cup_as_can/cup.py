# Run the LM-O single-object "can" model on the RealSense cup images (cup-as-can
# proxy). The cup has NO ground-truth pose and no cup-trained weights, so this is a
# PLUMBING / qualitative run: it proves custom RGB(-D) data flows through GDRNPP and
# produces poses + overlays. Expect rough alignment (cup != can; small/far bboxes).
#
# Inherits the SO can architecture (GDRN_double_mask, class-agnostic, 1 object) so the
# pretrained can checkpoint loads with no size mismatch. TRAIN is kept only so the
# renderer can build from the can CAD metadata; TEST is the custom cup dataset.
_base_ = "../lmoPbrSO/convnext_AugCosyAAEGray_DMask_amodalClipBox_lmo/ape.py"

OUTPUT_DIR = "output/gdrn/cup_as_can/ape_on_cup"

DATASETS = dict(
    TRAIN=("lmo_ape_train_pbr",),  # renderer metadata only (ape CAD); needs lmo models + train_pbr stub
    TEST=("custom_cup",),
    DET_FILES_TEST=("datasets/custom_data_1/cup_detections.json",),
    SYM_OBJS=[],
)

DATA_CFG = dict(custom_cup="configs/custom_data_1/cup.json")

INPUT = dict(WITH_DEPTH=False)

MODEL = dict(LOAD_DETS_TEST=True)

# No cup GT -> skip BOP eval entirely, just save the predicted-pose CSV.
TEST = dict(
    EVAL_PERIOD=0,
    VIS=False,
    USE_DEPTH_REFINE=False,
    SAVE_RESULTS_ONLY=True,
    TEST_BBOX_TYPE="est",
)
VAL = dict(SAVE_BOP_CSV_ONLY=True)
