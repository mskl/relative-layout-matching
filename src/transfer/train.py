#!/usr/bin/.env python3
import os

from core.settings import PROPRIETARY_DATASET_PATH, ELECTIONS_DATASET_PATH

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

import argparse
import datetime
import itertools
import logging
import os
import re
import sys

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

from core.callbacks import SegmentationImageCallback, TransferCallback
from core.datagen import PairDataset
from core.dataset import Dataset
from core.models import backbone_factory
from core.models.keras import PairModel
from core.train_utils import get_optimizer
from core.utils import load_clusters

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--consistency", default=0, type=float)
    parser.add_argument("--triplet", default=0, type=float)
    parser.add_argument("--reconstruction", default=0, type=float)
    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--batch-size", default=3, type=int)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--use-bos", default=0, type=int)
    parser.add_argument("--embdim", default=256, type=int)
    parser.add_argument("--transfer", default=1, type=int)
    parser.add_argument("--l2-normalize", default=1, type=int)
    parser.add_argument("--resume-saved", nargs="?")
    parser.add_argument("--tag", default="", type=str)
    parser.add_argument("--dataset", default="proprietary", choices=["proprietary", "elections"], type=str)
    parser.add_argument(
        "--contrastive-kind", choices=["triplet", "simclr"], default="triplet", type=str
    )
    parser.add_argument(
        "--consistency-kind", choices=["pairs", "variance"], default="pairs", type=str
    )
    backbones = [
        "vgg16", "resnet", "unet", "resnet_50_flat", "resnet_101_flat",
        "resnet_unet_50", "resnet_unet_101",
    ]
    parser.add_argument("--backbone", default="resnet_unet", choices=backbones, type=str)
    args = parser.parse_args([] if "__file__" not in globals() else None)

    args.threshold = 0.72 if args.backbone == "vgg16" else 0.5

    argstring = "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(
            ("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value)
             for key, value in sorted(vars(args).items())
             if key not in {"resume_saved"})
        )
    )

    args.logdir = os.path.join("/logs", argstring)

    if args.dataset == "proprietary":
        clusters_train = load_clusters("/data/clusters_v3_balanced_train.json")
        clusters_valid = load_clusters("/data/clusters_v3_balanced_valid.json")
        dataset_path = PROPRIETARY_DATASET_PATH
    else:
        clusters_train = load_clusters("/data/clusters_elections_train.json")
        clusters_valid = load_clusters("/data/clusters_elections_valid.json")
        dataset_path = ELECTIONS_DATASET_PATH

    documents_train = itertools.chain(*clusters_train)
    documents_valid = itertools.chain(*clusters_valid)

    dataset_train = Dataset(
        documents_train, preload_bos=args.use_bos, n_jobs=args.workers, dataset_path=dataset_path
    )
    dataset_valid = Dataset(
        documents_valid, preload_bos=args.use_bos, n_jobs=args.workers, dataset_path=dataset_path
    )

    c = clusters_valid[0]
    d1 = dataset_valid[c[0]]
    d2 = dataset_valid[c[1]]

    train_generator = PairDataset(
        dataset_train, clusters_train, batch_size=args.batch_size, include_unk=True,
        shuffle=True, backbone=args.backbone, include_bos=args.use_bos, n_repeats=4
    )
    valid_generator = PairDataset(
        dataset_valid, clusters_valid, batch_size=args.batch_size, include_unk=True,
        shuffle=False, backbone=args.backbone, include_bos=args.use_bos, n_repeats=1
    )

    backbone = backbone_factory(args.backbone, args.embdim, args.use_bos, args.transfer)

    tf.config.run_functions_eagerly(False)
    model = PairModel(
        backbone_model=backbone,
        consistency=args.consistency,
        triplet=args.triplet,
        reconstruction=args.reconstruction,
        contrastive_kind=args.contrastive_kind,
        consistency_kind=args.consistency_kind,
        l2_normalize=args.l2_normalize,
    )

    channels = 3 + (48 if args.use_bos else 0)
    model.build(input_shape=(None, 624, 880, channels))
    optimizer = get_optimizer(args.optimizer)
    if args.resume_saved:
        model.load_weights(f"/models/{args.resume_saved}")
    model.compile(optimizer=optimizer)

    multiprocess = {}
    if args.workers > 1:
        multiprocess = {
            "workers": args.workers,
            "max_queue_size": 10,
            "use_multiprocessing": True
        }

    writer = tf.summary.create_file_writer(args.logdir + "/train", flush_millis=2_000)
    callbacks = [
        TensorBoard(log_dir=args.logdir, profile_batch=0),
        SegmentationImageCallback(
            d1=d1,
            d2=d2,
            summary_writer=writer,
            include_bos=args.use_bos,
            backbone=args.backbone,
            eval_per_n_epochs=1,
        ),
        TransferCallback(
            generator=valid_generator,
            summary_writer=writer,
            argstring=argstring,
            threshold=args.threshold,
            eval_per_n_epochs=1,
        )
    ]

    model.fit(train_generator, epochs=args.epochs, callbacks=callbacks, **multiprocess)
    logger.info("Exiting finished training.")
    sys.exit(0)
