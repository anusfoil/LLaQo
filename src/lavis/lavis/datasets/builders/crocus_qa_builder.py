"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os

from lavis.common.registry import registry
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.crocus_qa import CROCUSDatasetQA


@registry.register_builder("crocus_qa")
class CROCUSQABuilder(BaseDatasetBuilder):
    train_dataset_cls = CROCUSDatasetQA
    eval_dataset_cls = CROCUSDatasetQA

    DATASET_CONFIG_DICT = {"default": "configs/datasets/crocus_qa/defaults.yaml"}

    def _download_ann(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        for split in build_info.splits:
            assert split in [
                "train",
                "val",
                "all",
            ], "Invalid split name {}, must be one of 'train', 'val' and 'all'."

            # eval_segments, balanced_train_segments, unbalanced_train_segments, all_train_segments
            seg_name = "train"  # TODO: fix the split name

            vis_processor = (
                self.vis_processors["train"]
                if split == "train"
                else self.vis_processors["eval"]
            )
            # create datasets
            dataset_cls = (
                self.eval_dataset_cls if split == "val" else self.train_dataset_cls
            )
            datasets[split] = dataset_cls(
                vis_processor=vis_processor,
                audio_root=build_info.storage,
                seg_name=seg_name,
            )

        return datasets

    def _download_data(self):
        pass
