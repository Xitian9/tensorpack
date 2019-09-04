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

__all__ = ['register_wood']


class WoodDetection(DatasetSplit):

    def __init__(self, basedir, name):
        """
        Args:
            basedir (str): root to the dataset
            name (str): the name of the split, e.g. "train2017"
        """
        basedir = os.path.expanduser(basedir)
        self.name = name
        self._imgdir = os.path.realpath(os.path.join(basedir, name))
        assert os.path.isdir(self._imgdir), "{} is not a directory!".format(self._imgdir)
        self.annotation_file = os.path.join(self._imgdir, 'groundTruth.csv')
        assert os.path.isfile(self.annotation_file), self.annotation_file

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
        image = {'file_name': ''}
        curBoxes = []
        curClasses = []

        def appendRoidb(img):
            if img["file_name"]:
                img["image_id"] = img["file_name"]
                self._use_absolute_file_name(img)
                if add_gt:
                    img["boxes"] = np.array(curBoxes)
                    img["class"] = np.array(curClasses)
                    img["is_crowd"] = np.array([False] * len(curBoxes))
                roidbs.append(img)

        with timed_operation('Load Groundtruth Boxes for {}'.format(self.name)):

            df = pd.read_csv(self.annotation_file,
                             names=["filename", "x1", "y1", "x2", "y2", "class", "area"])

            for index, row in df.iterrows():
                fn = row["filename"]
                if fn != image["file_name"]:
                    appendRoidb(image)
                    image = {'file_name': fn}
                    curBoxes = []
                    curClasses = []

                curBoxes.append([np.float32(row["x1"]), np.float32(row["y1"]),
                                 np.float32(row["x2"]), np.float32(row["y2"])])
                if row["class"] == "1":
                    if row["area"] < 200:
                        curClasses.append(1)
                    elif row["area"] < 350:
                        curClasses.append(2)
                    else:
                        curClasses.append(3)
                elif row["class"] == "2":
                    if row["area"] < 22:
                        curClasses.append(4)
                    elif row["area"] < 30:
                        curClasses.append(5)
                    else:
                        curClasses.append(6)

                appendRoidb(image)

        return roidbs

    def _use_absolute_file_name(self, img):
        """
        Change relative filename to abosolute file name.
        """
        img['file_name'] = os.path.join(
            self._imgdir, img['file_name'])
        assert os.path.isfile(img['file_name']), img['file_name']

    def training_roidbs(self):
        return self.load(add_gt=True, add_mask=cfg.MODE_MASK)

    def inference_roidbs(self):
        return self.load(add_gt=False)

    def eval_inference_results(self, results, output):
        assert output is not None, "evaluation requires an output file!"
        with open(output, 'w') as f:
            json.dump(results, f)
        if len(results):
            # sometimes may crash if the results are empty?
            #return results
            # Not working at the moment: need to set up proper data summaries
            return {}
        else:
            return {}


def register_wood(basedir):
    """
    Add COCO datasets like "coco_train201x" to the registry,
    so you can refer to them with names in `cfg.DATA.TRAIN/VAL`.
    """
    class_names = ["small_hole", "medium_hole", "large_hole",
                   "small_branch", "medium_branch", "large_branch"]
    class_names = ["BG"] + class_names


    for split in ["train", "val"]:
        name = "wood_" + split
        DatasetRegistry.register(name, lambda x=split: WoodDetection(basedir, x))
        DatasetRegistry.register_metadata(name, 'class_names', class_names)
