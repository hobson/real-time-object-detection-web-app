"""train_resume_all_categories.KITTI_TO_UNIFIED must have an entry for
every one of KITTI's own 8 class ids (0-7 per data/external_datasets/
kitti/kitti.yaml) - db_load_training_images.py and train_resume_all_
categories.py both do a plain dict lookup (`KITTI_TO_UNIFIED[kitti_cid]`),
which raises KeyError on any label file containing a class id this dict
doesn't cover, rather than silently dropping it - so "the mapping is
total over KITTI's real class range" is a correctness invariant worth
locking in, not just an implementation detail."""
from train_resume_all_categories import KITTI_TO_UNIFIED

KITTI_CLASS_IDS = range(8)  # car, van, truck, pedestrian, Person_sitting, cyclist, tram, misc


def test_every_kitti_class_id_has_an_entry():
    missing = [cid for cid in KITTI_CLASS_IDS if cid not in KITTI_TO_UNIFIED]
    assert missing == []


def test_mapped_values_are_valid_unified_ids_or_none():
    # Unified scheme is COCO's 0-79 plus license_plate=80 - a remap target
    # outside that range would silently corrupt training data (wrong class
    # name) rather than erroring the way a KeyError would.
    for cid, unified in KITTI_TO_UNIFIED.items():
        assert unified is None or 0 <= unified <= 80


def test_misc_is_explicitly_dropped():
    # KITTI class 7 ("misc") has no reasonable single-class match (see the
    # module's own inline comment) - must map to None (explicit drop), not
    # accidentally collapse onto some other class's id.
    assert KITTI_TO_UNIFIED[7] is None
