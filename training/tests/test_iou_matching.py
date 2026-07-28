"""review_confusion_matrix.py's iou()/match_boxes() are reused by
compute_class_fn_fp_rates.py, compute_label_confidence.py, and
flag_review_images.py - previously untested despite being load-bearing for
all of them."""
from review_confusion_matrix import iou, match_boxes


def test_iou_identical_boxes_is_one():
    box = (0.1, 0.1, 0.5, 0.5)
    assert iou(box, box) == 1.0


def test_iou_disjoint_boxes_is_zero():
    assert iou((0.0, 0.0, 0.1, 0.1), (0.5, 0.5, 0.6, 0.6)) == 0.0


def test_iou_partial_overlap():
    # Two unit-ish boxes overlapping in a 0.5x0.5 region out of a
    # 1.5x1 union - exact known ratio, not just "somewhere between 0 and 1".
    a = (0.0, 0.0, 1.0, 1.0)
    b = (0.5, 0.0, 1.5, 1.0)
    # intersection: 0.5x1.0 = 0.5; union: 1.0 + 1.0 - 0.5 = 1.5
    assert iou(a, b) == 0.5 / 1.5


def test_iou_zero_area_box_is_zero_not_nan():
    # A degenerate (zero-width) box must not divide-by-zero into NaN/inf -
    # union area is 0 in this case, and iou() explicitly guards that.
    assert iou((0.1, 0.1, 0.1, 0.5), (0.1, 0.1, 0.1, 0.5)) == 0.0


def test_match_boxes_greedy_prefers_higher_iou():
    # Two ground-truth boxes, two predictions - pred[0] overlaps both gt
    # boxes but more with gt[1]; greedy-by-descending-score matching should
    # assign pred[0] to gt[1] (its best match) and leave gt[0] unmatched
    # since there's no other prediction near it.
    gt = [(0.0, 0.0, 0.2, 0.2), (0.0, 0.0, 1.0, 1.0)]
    preds = [(0.05, 0.05, 0.95, 0.95)]
    matched_gt, matched_pred = match_boxes(gt, preds, iou_thresh=0.5)
    assert matched_gt == {1}
    assert matched_pred == {0}


def test_match_boxes_below_threshold_unmatched():
    gt = [(0.0, 0.0, 0.1, 0.1)]
    preds = [(0.5, 0.5, 0.6, 0.6)]
    matched_gt, matched_pred = match_boxes(gt, preds, iou_thresh=0.5)
    assert matched_gt == set()
    assert matched_pred == set()
