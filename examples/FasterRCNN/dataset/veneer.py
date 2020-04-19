# -*- coding: utf-8 -*-

import json
import os
import numpy as np
import pandas as pd
import tqdm


from tensorpack.utils import logger
from tensorpack.utils.timer import timed_operation

from config import config as cfg
from dataset import DatasetRegistry, DatasetSplit

__all__ = ['register_veneer']


class VeneerDetection(DatasetSplit):

    def __init__(self, basedir, annotation_file, name):
        """
        Args:
            basedir (str): root to the dataset
            annotation_file (str): ground truth file
            name (str): the name of the split, e.g. "train2017"
        """
        self.basedir = os.path.expanduser(basedir)
        self.name = name
        self.annotation_file = os.path.realpath(os.path.join(self.basedir, annotation_file))
        assert os.path.isfile(self.annotation_file), self.annotation_file
        self._imgdir = os.path.realpath(os.path.join(self.basedir, name))
        # assert os.path.isdir(self._imgdir), "{} is not a directory!".format(self._imgdir)

        logger.info("Instances loaded from {}.".format(self.annotation_file))

    def load(self, add_gt=True, add_mask=False):
        """
        Args:
            add_gt: whether to add ground truth bounding box annotations to the dicts
            add_mask: whether to also add ground truth mask

        Returns:
            a list of dict, each has keys including:
                'image_id', 'file_name',
                and (if add_gt is True) 'boxes', 'class', 'is_crowd', and optionally
                'segmentation'.
        """
        assert not add_mask

        roidbs = []
        image = {'dataset': '', 'file_name': ''}
        curBoxes = []
        curClasses = []

        def appendRoidb(img):
            dataset = image["dataset"]
            filename = image["file_name"]

            base = os.path.join(self.basedir, dataset)
            splits = [f for f in os.scandir(base)
                          if f.name == self.name
                             and f.is_dir()
                             and os.path.isfile(os.path.join(f.path, filename))]

            if len(splits) == 1 and filename:
                img["file_name"] = os.path.join(splits[0].path, filename)
                img["image_id"] = os.path.join(dataset, filename)
                self._use_absolute_file_name(img)
                if add_gt:
                    img["boxes"] = np.array(curBoxes)
                    img["class"] = np.array(curClasses)
                    img["is_crowd"] = np.array([False] * len(curBoxes))
                roidbs.append(img)

        with timed_operation('Load Groundtruth Boxes for {}'.format(self.name)):

            df = pd.read_csv(self.annotation_file,
                             names=["x", "y", "width", "height", "class",
                                    "label", "dataset", "filename"])

            for index, row in df.iterrows():
                fn = row["filename"]
                dn = row["dataset"]
                if not(dn == image["dataset"] and fn == image["file_name"]):
                    appendRoidb(image)
                    image = {'dataset': dn, 'file_name': fn}
                    curBoxes = []
                    curClasses = []

                x1 = np.float32(row["x"])
                y1 = np.float32(row["y"])
                width = row["width"]
                height =row["height"]
                x2 = np.float32(x1 + width)
                y2 = np.float32(y1 + height)

                curBoxes.append([x1,y1,x2,y2])
                curClasses.append(row["class"])
                appendRoidb(image)

        return roidbs

    def _use_absolute_file_name(self, img):
        """
        Change relative filename to abosolute file name.
        """
        img['file_name'] = os.path.join(
            self._imgdir, img['dataset'], img['file_name'])
        assert os.path.isfile(img['file_name']), img['file_name']

    def training_roidbs(self):
        return self.load(add_gt=True, add_mask=cfg.MODE_MASK)

    def inference_roidbs(self):
        return self.load(add_gt=False)

    def eval_inference_results(self, results, output=None):
        if output is not None:
            with open(output, 'w') as f:
                json.dump(results, f)
        if len(results):
            # sometimes may crash if the results are empty?
            #return results
            # Not working at the moment: need to set up proper data summaries
            return {}
        else:
            return {}


def register_veneer(basedir, groundTruth):
    """
    Add COCO datasets like "coco_train201x" to the registry,
    so you can refer to them with names in `cfg.DATA.TRAIN/VAL`.
    """
    class_names = ["Bad Branch", "Hole", "Crack", "End Defect", "Good Branch", "Rot"]
    class_names = ["BG"] + class_names

    df = pd.read_csv(groundTruth,
                     names=["x", "y", "width", "height", "class", "label", "dataset", "filename"])

    for dataset in set(row["dataset"] for index, row in df.iterrows()):
        for split in ["train", "val"]:
            name = dataset + "_" + split
            DatasetRegistry.register(name, lambda x=split: VeneerDetection(basedir, groundTruth, x))
            DatasetRegistry.register_metadata(name, 'class_names', class_names)
