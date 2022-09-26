# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash
"""Generate labeled and unlabeled dataset for coco train.

Example:
python tools/coco_semi.py
"""

import argparse
import numpy as np
import json
import os


def prepare_voc_data(seed=1, percent=50.0, version='07', seed_offset=0):
    """Prepare VOC dataset for the Selective Semi-Supervised Learning
    Args:
      seed: random seed for dataset split
      percent: percentage of labeled dataset
      version: VOC dataset version
    """

    def _save_anno(name, images, annotations):
        """Save annotation."""
        print(
            ">> Processing dataset {}.json saved ({} images {} annotations)".format(
                name, len(images), len(annotations)
            )
        )
        new_anno = {}
        new_anno["images"] = images
        new_anno["annotations"] = annotations
        # new_anno["licenses"] = anno["licenses"]
        new_anno["categories"] = anno["categories"]
        # new_anno["info"] = anno["info"]
        path = "{}/{}".format(VOCANNODIR, "selective")
        if not os.path.exists(path):
            os.mkdir(path)

        with open(
            "{root}/{folder}/{save_name}.json".format(
                save_name=name, root=VOCANNODIR, folder="selective"
            ),
            "w",
        ) as f:
            json.dump(new_anno, f)
        print(
            ">> Data {}.json saved ({} images {} annotations)".format(
                name, len(images), len(annotations)
            )
        )

    np.random.seed(seed + seed_offset)
    VOCANNODIR = os.path.join(DATA_DIR, "annos_coco")

    anno = json.load(
        open(os.path.join(VOCANNODIR, "voc{}_trainval.json".format(version)))
    )

    image_list = anno["images"]
    labeled_tot = int(percent / 100.0 * len(image_list))
    labeled_ind = np.random.choice(
        range(len(image_list)), size=labeled_tot, replace=False
    )
    labeled_id = []
    labeled_images = []
    selective_images = []
    labeled_ind = set(labeled_ind)
    for i in range(len(image_list)):
        if i in labeled_ind:
            labeled_images.append(image_list[i])
            labeled_id.append(image_list[i]["id"])
        else:
            selective_images.append(image_list[i])

    # get all annotations of labeled images
    labeled_id = set(labeled_id)
    labeled_annotations = []
    selective_annotations = []
    for an in anno["annotations"]:
        if an["image_id"] in labeled_id:
            labeled_annotations.append(an)
        else:
            selective_annotations.append(an)

    # save labeled and selective
    save_name = "voc{version}_trainval.{seed}@{tot}".format(
        version=version, seed=seed, tot=int(percent)
    )
    _save_anno(save_name, labeled_images, labeled_annotations)
    save_name = "voc{version}_trainval.{seed}@{tot}-selective".format(
        version=version, seed=seed, tot=int(percent)
    )
    _save_anno(save_name, selective_images, selective_annotations)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--percent", type=float, default=50)
    parser.add_argument("--version", type=str, default='07')
    parser.add_argument("--seed", type=int, help="seed", default=1)
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()
    print(args)
    DATA_DIR = args.data_dir
    prepare_voc_data(args.seed, args.percent, args.version, args.seed_offset)
