"""
Author: Yawei Li
Email: yawei.li@vision.ee.ethz.ch

Count the number of images in each subset.
"""


import glob
import os
import argparse
from collections import defaultdict
from pprint import pprint
from clean_images import query_sets, l_partitions, file_types

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Count number of images.')
    parser.add_argument('--save_dir', default='../flickr_test', type=str)
    # /cluster/work/cvl/videosr/flickr_data
    args = parser.parse_args()

    num_images = defaultdict(list)

    for q_set in query_sets:
        for key, (folder, regex, _) in file_types.items():
            for lp in l_partitions:
                save_dir = os.path.join(args.save_dir, q_set, folder, lp, regex)
                img_names = glob.glob(save_dir)
                num_images[f"{q_set}_{key}"].append(len(img_names))
            num_images[f"{q_set}_{key}"].append(sum(num_images[f"{q_set}_{key}"]))

    pprint(num_images)


