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

from dataset import register_veneer
from config import config as base_config
from config import finalize_configs
from config_list import config_list
import copy
from data import get_train_dataflow
from eval import EvalCallback
from modeling.generalized_rcnn import ResNetC4Model, ResNetFPNModel


try:
    import horovod.tensorflow as hvd
except ImportError:
    pass


def run_training(cfg, logdir, load=None):

    for gt in cfg.DATA.GROUNDTRUTHS:
        register_veneer(cfg.DATA.BASEDIR, gt)  # add datasets to the registry

    # Setup logging ...
    is_horovod = cfg.TRAINER == 'horovod'
    if is_horovod:
        hvd.init()
    if not is_horovod or hvd.rank() == 0:
        logger.set_logger_dir(logdir, 'd')
    logger.info("Environment Information:\n" + collect_env_info())

    finalize_configs(cfg, is_training=True)

    # Create model
    MODEL = ResNetFPNModel() if base_config.MODE_FPN else ResNetC4Model()

    # Compute the training schedule from the number of GPUs ...
    stepnum = cfg.TRAIN.STEPS_PER_EPOCH
    # warmup is step based, lr is epoch based
    init_lr = cfg.TRAIN.WARMUP_INIT_LR * min(8. / cfg.TRAIN.NUM_GPUS, 1.)
    warmup_schedule = [(0, init_lr), (cfg.TRAIN.WARMUP, cfg.TRAIN.BASE_LR)]
    warmup_end_epoch = cfg.TRAIN.WARMUP * 1. / stepnum
    lr_schedule = [(int(warmup_end_epoch + 0.5), cfg.TRAIN.BASE_LR)]

    factor = 8. / cfg.TRAIN.NUM_GPUS
    for idx, steps in enumerate(cfg.TRAIN.LR_SCHEDULE[:-1]):
        mult = 0.1 ** (idx + 1)
        lr_schedule.append(
            (steps * factor // stepnum, cfg.TRAIN.BASE_LR * mult))
    logger.info("Warm Up Schedule (steps, value): " + str(warmup_schedule))
    logger.info("LR Schedule (epochs, value): " + str(lr_schedule))
    train_dataflow = get_train_dataflow()
    # This is what's commonly referred to as "epochs"
    total_passes = cfg.TRAIN.LR_SCHEDULE[-1] * 8 / train_dataflow.size()
    logger.info("Total passes of the training set is: {:.5g}".format(total_passes))

    # Create callbacks ...
    callbacks = [
        PeriodicCallback(
            ModelSaver(max_to_keep=10, keep_checkpoint_every_n_hours=1),
            every_k_epochs=cfg.TRAIN.CHECKPOINT_PERIOD),
        # linear warmup
        ScheduledHyperParamSetter(
            'learning_rate', warmup_schedule, interp='linear', step_based=True),
        ScheduledHyperParamSetter('learning_rate', lr_schedule),
        HostMemoryTracker(),
        ThroughputTracker(samples_per_step=cfg.TRAIN.NUM_GPUS),
        EstimatedTimeLeft(median=True),
        SessionRunTimeout(60000),   # 1 minute timeout
        GPUUtilizationTracker()
    ]
    if cfg.TRAIN.EVAL_PERIOD > 0:
        callbacks.extend([
            EvalCallback(dataset, *MODEL.get_inference_tensor_names(), args.logdir)
            for dataset in cfg.DATA.VAL
        ])

    if cfg.TRAINER != 'cpu':
        callbacks.append(GPUMemoryTracker())

    if cfg.TRAINER == 'replicated' and platform != 'win32':
        callbacks.append(GPUUtilizationTracker())

    if is_horovod and hvd.rank() > 0:
        session_init = None
    else:
        if args.load:
            # ignore mismatched values, so you can `--load` a model for fine-tuning
            session_init = SmartInit(args.load, ignore_mismatch=True)
        else:
            session_init = SmartInit(cfg.BACKBONE.WEIGHTS)

    traincfg = TrainConfig(
        model=MODEL,
        data=QueueInput(train_dataflow),
        callbacks=callbacks,
        steps_per_epoch=stepnum,
        max_epoch=cfg.TRAIN.LR_SCHEDULE[-1] * factor // stepnum,
        session_init=session_init,
        starting_epoch=cfg.TRAIN.STARTING_EPOCH
    )

    if is_horovod:
        trainer = HorovodTrainer(average=False)
    elif cfg.TRAINER == 'replicated':
        # nccl mode appears faster than cpu mode
        trainer = SyncMultiGPUTrainerReplicated(cfg.TRAIN.NUM_GPUS, average=False, mode='nccl')
    else:
        trainer = QueueInputTrainer()
    launch_train_with_config(traincfg, trainer)

if __name__ == '__main__':
    # "spawn/forkserver" is safer than the default "fork" method and
    # produce more deterministic behavior & memory saving
    # However its limitation is you cannot pass a lambda function to subprocesses.
    import multiprocessing as mp
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()

    if get_tf_version_tuple() < (1, 6):
        # https://github.com/tensorflow/tensorflow/issues/14657
        logger.warn("TF<1.6 has a bug which may lead to crash in FasterRCNN if you're unlucky.")

    for config in config_list:
        cfg, logdir, load = config
        run_training(cfg, logdir, load)
