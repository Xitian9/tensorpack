#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: train.py

import argparse
from sys import platform
import six
assert six.PY3, "This example requires Python 3!"

from tensorpack import *
from tensorpack.callbacks.prof import *
from tensorpack.tfutils import collect_env_info
from tensorpack.tfutils.common import get_tf_version_tuple

from dataset import register_coco, register_shapes, register_wood
from config import config as cfg
from config import finalize_configs
from data import get_train_dataflow
from eval import EvalCallback
from modeling.generalized_rcnn import ResNetC4Model, ResNetFPNModel


try:
    import horovod.tensorflow as hvd
except ImportError:
    pass


if __name__ == '__main__':
    # "spawn/forkserver" is safer than the default "fork" method and
    # produce more deterministic behavior & memory saving
    # However its limitation is you cannot pass a lambda function to subprocesses.
    import multiprocessing as mp
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', help='load a model to start training from. Can overwrite BACKBONE.WEIGHTS')
    parser.add_argument('--logdir', help='log directory', default='train_log/maskrcnn')
    parser.add_argument('--config', help="A list of KEY=VALUE to overwrite those defined in config.py", nargs='+')

    if get_tf_version_tuple() < (1, 6):
        # https://github.com/tensorflow/tensorflow/issues/14657
        logger.warn("TF<1.6 has a bug which may lead to crash in FasterRCNN if you're unlucky.")

    args = parser.parse_args()
    if args.config:
        cfg.update_args(args.config)
    # register_wood(cfg.DATA.BASEDIR)  # add datasets to the registry

    finalize_configs(is_training=True)

    # Create model
    MODEL = ResNetFPNModel() if cfg.MODE_FPN else ResNetC4Model()

    if args.load:
        session_init = get_model_loader(args.load)
    else:
        session_init = get_model_loader(cfg.BACKBONE.WEIGHTS) if cfg.BACKBONE.WEIGHTS else None

    with session_init as sess:
        print([m.values() for m in graph.get_operations()])

    #if is_horovod and hvd.rank() > 0:
    #    session_init = None
    #else:
    #    if args.load:
    #        session_init = get_model_loader(args.load)
    #    else:
    #        session_init = get_model_loader(cfg.BACKBONE.WEIGHTS) if cfg.BACKBONE.WEIGHTS else None

    #traincfg = TrainConfig(
    #    model=MODEL,
    #    data=QueueInput(train_dataflow),
    #    callbacks=callbacks,
    #    steps_per_epoch=stepnum,
    #    max_epoch=cfg.TRAIN.LR_SCHEDULE[-1] * factor // stepnum,
    #    session_init=session_init,
    #    starting_epoch=cfg.TRAIN.STARTING_EPOCH
    #)
    #if is_horovod:
    #    trainer = HorovodTrainer(average=False)
    #elif cfg.TRAINER == 'replicated' and cfg.TRAIN.NUM_GPUS > 0:
    #    # nccl mode appears faster than cpu mode
    #    trainer = SyncMultiGPUTrainerReplicated(cfg.TRAIN.NUM_GPUS, average=False, mode='nccl')
    #else:
    #    trainer = QueueInputTrainer()
    #launch_train_with_config(traincfg, trainer)
