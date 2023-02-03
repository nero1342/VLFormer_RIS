# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/sukjunhwang/IFC

import contextlib
import copy
import io
import itertools
import json
import logging
from re import S
import numpy as np
import os
from collections import OrderedDict
import pycocotools.mask as mask_util
import torch
# from .datasets.ref_ytvos_api.ytvos import YTVOS
from tabulate import tabulate

import detectron2.utils.comm as comm
from detectron2.config import CfgNode
from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluator
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import create_small_table

class YTVISEvaluator(DatasetEvaluator):
    """
    Evaluate AR for object proposals, AP for instance detection/segmentation, AP
    for keypoint detection outputs using COCO's metrics.
    See http://cocodataset.org/#detection-eval and
    http://cocodataset.org/#keypoints-eval to understand its metrics.

    In addition to COCO, this evaluator is able to support any bounding box detection,
    instance segmentation, or keypoint detection dataset.
    """

    def __init__(
        self,
        dataset_name,
        tasks=None,
        distributed=True,
        output_dir=None,
        *,
        use_fast_impl=True,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:

                    "json_file": the path to the COCO format annotation

                Or it must be in detectron2's standard dataset format
                so it can be converted to COCO format automatically.
            tasks (tuple[str]): tasks that can be evaluated under the given
                configuration. A task is one of "bbox", "segm", "keypoints".
                By default, will infer this automatically from predictions.
            distributed (True): if True, will collect results from all ranks and run evaluation
                in the main process.
                Otherwise, will only evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains two files:

                1. "instances_predictions.pth" a file in torch serialization
                   format that contains all the raw original predictions.
                2. "coco_instances_results.json" a json file in COCO's result
                   format.
            use_fast_impl (bool): use a fast but **unofficial** implementation to compute AP.
                Although the results should be very close to the official implementation in COCO
                API, it is still recommended to compute results with the official API for use in
                papers. The faster implementation also uses more RAM.
        """
        self._logger = logging.getLogger(__name__)
        self._distributed = distributed
        self._output_dir = output_dir
        self._use_fast_impl = use_fast_impl

        if tasks is not None and isinstance(tasks, CfgNode):
            self._logger.warning(
                "COCO Evaluator instantiated using config, this is deprecated behavior."
                " Please pass in explicit arguments instead."
            )
            self._tasks = None  # Infering it from predictions should be better
        else:
            self._tasks = tasks

        self._cpu_device = torch.device("cpu")

        self._metadata = MetadataCatalog.get(dataset_name)
        from pycocotools.coco import COCO

        json_file = PathManager.get_local_path(self._metadata.json_file)
        
        with contextlib.redirect_stdout(io.StringIO()):
            self._ytvis_api = COCO(json_file)
        
        # with contextlib.redirect_stdout(io.StringIO()):
        #     self._ytvis_api = YTVOS(json_file)

        # Test set json files do not contain annotations (evaluation must be
        # performed using the COCO evaluation server).
        self._do_evaluation = "annotations" in self._ytvis_api.dataset

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        prediction = instances_to_coco_json_video(inputs, outputs)
        # if self._output_dir:
        #     for i, p in enumerate(prediction):
        #         self.save_single_output(self._output_dir, p, i)

        for i, p in enumerate(prediction):
            pred = p["segmentations"]
            # gt = self._ytvis_api.loadAnnsFromExps(p["id"])[0]
            try:
                ann = self._ytvis_api.imgToAnns[int(p["video_id"])][0]
                gt = self._ytvis_api.annToMask(ann)[None,]
                p["iou"] = db_eval_iou(pred, gt)
                p["boundary"] = db_eval_boundary(pred, gt)
                # p["segmentations"] = gt 
            except:
                p["iou"] = 0
                p["boundary"] = 0
            

            # self.save_single_output(self._output_dir, p, i)
            del p["segmentations"]
        self._predictions.extend(prediction)

    def evaluate(self):
        """
        Args:
            img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset
        """
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        self._results = OrderedDict()
        self._eval_predictions(predictions)
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _eval_predictions(self, predictions):
        """
        Evaluate predictions. Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for YTVIS format ...")

        # if self._output_dir:
        #     file_path = os.path.join(self._output_dir, "Annotations")
        #     self._logger.info("Saving results to {}".format(file_path))
        #     self.save_output(file_path, predictions)

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return
        
        self._logger.info("Evaluating...")
            
        iou = [] 
        boundary = [] 
        for prediction in predictions:
            iou.extend(prediction["iou"])
            boundary.extend(prediction["boundary"]) 

        self._results["iou"] = np.mean(np.array(iou))
        self._results["f1"] = np.mean(np.array(boundary))
        self._results["overall"] = (self._results["iou"] + self._results["f1"]) / 2
        # Precision@X
        for thres in range(5, 10):
            self._results[f"Pr@{thres * 10}"] = np.mean(np.array(iou) > thres / 10)

    def save_single_output(self, root, prediction, obj_id):
        
        from PIL import Image 
        video_id = prediction["video_id"]
        exp_id = prediction["exp_id"]
        file_names = prediction["file_names"]
        masks = prediction["segmentations"]
        exp = prediction["expression"]
        # dir = os.path.join(root, video_id, f"{exp_id}_{exp}_{iou}_{f1}")
        for i, file_name in enumerate(file_names):
            iou = prediction["iou"][i]
            f1 = prediction["boundary"][i]
            x = file_name.split('/')[-1].split('.')[0]
            dir = os.path.join(root, x)
            os.makedirs(dir, exist_ok=True)
            mask = Image.fromarray(masks[i] * 255).convert("P")
            exp = exp.replace("/", " ")
            # mask.save(os.path.join(dir, file_name.split('/')[-1]).replace('.jpg', '.png'))
            mask.save(os.path.join(dir, f"{exp}_{iou}_{f1}_{video_id}_{exp_id}.png"))
def instances_to_coco_json_video(inputs, outputs):
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.

    Args:
        instances (Instances):
        video_id (int): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    assert len(inputs) == 1, "More than one inputs are loaded for inference!"

    video_id = inputs[0]["video_name"]
    video_length = inputs[0]["length"]
    exp = inputs[0]["expression"]
    file_names = inputs[0]["file_names"]
    exp_id = inputs[0]["exp_id"]
    id = inputs[0]["id"]

    scores = outputs["pred_scores"]
    labels = outputs["pred_labels"]
    masks = outputs["pred_masks"]

    ytvis_results = []
    for instance_id, (s, l, m) in enumerate(zip(scores, labels, masks)):
        m = np.array(m).astype(np.uint8)
        res = {
            "id": id, 
            "file_names": file_names,
            "expression": exp,
            "video_id": video_id,
            "exp_id": exp_id,
            "score": s,
            "category_id": l,
            "segmentations": m,
        }
        ytvis_results.append(res)

    return ytvis_results


import math
import numpy as np
import cv2


def db_eval_iou(annotation, segmentation, void_pixels=None):
    """ Compute region similarity as the Jaccard Index.
    Arguments:
        annotation   (ndarray): binary annotation   map.
        segmentation (ndarray): binary segmentation map.
        void_pixels  (ndarray): optional mask with void pixels
    Return:
        jaccard (float): region similarity
    """
    assert annotation.shape == segmentation.shape, \
        f'Annotation({annotation.shape}) and segmentation:{segmentation.shape} dimensions do not match.'
    annotation = annotation.astype(np.bool)
    segmentation = segmentation.astype(np.bool)

    if void_pixels is not None:
        assert annotation.shape == void_pixels.shape, \
            f'Annotation({annotation.shape}) and void pixels:{void_pixels.shape} dimensions do not match.'
        void_pixels = void_pixels.astype(np.bool)
    else:
        void_pixels = np.zeros_like(segmentation)

    # Intersection between all sets
    inters = np.sum((segmentation & annotation) & np.logical_not(void_pixels), axis=(-2, -1))
    union = np.sum((segmentation | annotation) & np.logical_not(void_pixels), axis=(-2, -1))

    j = inters / union
    if j.ndim == 0:
        j = 1 if np.isclose(union, 0) else j
    else:
        j[np.isclose(union, 0)] = 1
    return j


def db_eval_boundary(annotation, segmentation, void_pixels=None, bound_th=0.008):
    assert annotation.shape == segmentation.shape
    if void_pixels is not None:
        assert annotation.shape == void_pixels.shape
    if annotation.ndim == 3:
        n_frames = annotation.shape[0]
        f_res = np.zeros(n_frames)
        for frame_id in range(n_frames):
            void_pixels_frame = None if void_pixels is None else void_pixels[frame_id, :, :, ]
            f_res[frame_id] = f_measure(segmentation[frame_id, :, :, ], annotation[frame_id, :, :], void_pixels_frame, bound_th=bound_th)
    elif annotation.ndim == 2:
        f_res = f_measure(segmentation, annotation, void_pixels, bound_th=bound_th)
    else:
        raise ValueError(f'db_eval_boundary does not support tensors with {annotation.ndim} dimensions')
    return f_res


def f_measure(foreground_mask, gt_mask, void_pixels=None, bound_th=0.008):
    """
    Compute mean,recall and decay from per-frame evaluation.
    Calculates precision/recall for boundaries between foreground_mask and
    gt_mask using morphological operators to speed it up.
    Arguments:
        foreground_mask (ndarray): binary segmentation image.
        gt_mask         (ndarray): binary annotated image.
        void_pixels     (ndarray): optional mask with void pixels
    Returns:
        F (float): boundaries F-measure
    """
    assert np.atleast_3d(foreground_mask).shape[2] == 1
    if void_pixels is not None:
        void_pixels = void_pixels.astype(np.bool)
    else:
        void_pixels = np.zeros_like(foreground_mask).astype(np.bool)

    bound_pix = bound_th if bound_th >= 1 else \
        np.ceil(bound_th * np.linalg.norm(foreground_mask.shape))

    # Get the pixel boundaries of both masks
    fg_boundary = _seg2bmap(foreground_mask * np.logical_not(void_pixels))
    gt_boundary = _seg2bmap(gt_mask * np.logical_not(void_pixels))

    from skimage.morphology import disk

    # fg_dil = binary_dilation(fg_boundary, disk(bound_pix))
    fg_dil = cv2.dilate(fg_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))
    # gt_dil = binary_dilation(gt_boundary, disk(bound_pix))
    gt_dil = cv2.dilate(gt_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))

    # Get the intersection
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    # Area of the intersection
    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)

    # % Compute precision and recall
    if n_fg == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        F = 0
    else:
        F = 2 * precision * recall / (precision + recall)

    return F


def _seg2bmap(seg, width=None, height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.  The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.
    Arguments:
        seg     : Segments labeled from 1..k.
        width	  :	Width of desired bmap  <= seg.shape[1]
        height  :	Height of desired bmap <= seg.shape[0]
    Returns:
        bmap (ndarray):	Binary boundary map.
     David Martin <dmartin@eecs.berkeley.edu>
     January 2003
    """

    seg = seg.astype(np.bool)
    seg[seg > 0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height

    h, w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (
        width > w | height > h | abs(ar1 - ar2) > 0.01
    ), "Can" "t convert %dx%d seg to %dx%d bmap." % (w, h, width, height)

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height, width))
        for x in range(w):
            for y in range(h):
                if b[y, x]:
                    j = 1 + math.floor((y - 1) + height / h)
                    i = 1 + math.floor((x - 1) + width / h)
                    bmap[j, i] = 1

    return bmap