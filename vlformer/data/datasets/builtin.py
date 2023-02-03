# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/sukjunhwang/IFC

import os


from .ref_ytvos import (
    register_ref_ytvos_instances,
    _get_ref_ytvos_2021_instances_meta,
)

from .coco import register_coco_instances

# ==== Predefined splits for RefYTVIS 2021 ===========
_PREDEFINED_SPLITS_REF_YTVOS_2021 = {
    "ref_ytvos_2021_train": ("ref_ytvos_2021/train/JPEGImages",
                         "ref_ytvos_2021/train.json"),
    "ref_ytvos_2021_val_fake": ("ref_ytvos_2021/train/JPEGImages",
                         "ref_ytvos_2021/valid_fake.json"),
    "ref_ytvos_2021_val": ("ref_ytvos_2021/valid/JPEGImages",
                       "ref_ytvos_2021/valid.json"),
    "ref_ytvos_2021_test": ("ref_ytvos_2021/test/JPEGImages",
                        "ref_ytvos_2021/test.json"),
}

# ==== Predefined splits for RefCoco 2014 ===========
_PREDEFINED_SPLITS_REF_COCO_2014 = {
    "refcoco_train": ("coco/train2014",
                         "coco/refcoco/instances_refcoco_train.json"),
    "refcoco_val": ("coco/train2014",
                         "coco/refcoco/instances_refcoco_val.json"),
    "refcoco_testA": ("coco/train2014",
                         "coco/refcoco/instances_refcoco_testA.json"),
    "refcoco_testA_fake": ("coco/train2014",
                         "coco/refcoco/instances_refcoco_testA_fake.json"),
    "refcoco_testB": ("coco/train2014",
                         "coco/refcoco/instances_refcoco_testB.json"),                                          
    "refcoco+_train": ("coco/train2014",
                        "coco/refcoco+/instances_refcoco+_train.json"),
    "refcoco+_val": ("coco/train2014",
                        "coco/refcoco+/instances_refcoco+_val.json"),
    "refcoco+_testA": ("coco/train2014",
                        "coco/refcoco+/instances_refcoco+_testA.json"),
    "refcoco+_testB": ("coco/train2014",
                        "coco/refcoco+/instances_refcoco+_testB.json"),
    "refcocog_train": ("coco/train2014",
                         "coco/refcocog/instances_refcocog_train.json"),
    "refcocog_val": ("coco/train2014",
                         "coco/refcocog/instances_refcocog_val.json"),
    "refcocog_test": ("coco/train2014",
                         "coco/refcocog/instances_refcocog_test.json"),
}


def register_all_ref_ytvos_2021(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_REF_YTVOS_2021.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ref_ytvos_instances(
            key,
            _get_ref_ytvos_2021_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

def register_all_ref_coco_2014(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_REF_COCO_2014.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            {},
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )



if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    register_all_ref_ytvos_2021(_root)
    register_all_ref_coco_2014(_root)