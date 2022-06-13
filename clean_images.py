"""
Author: Yawei Li
Email: yawei.li@vision.ee.ethz.ch

Clean the images after running flickrapi_downloader_threading.py
    1. Remove inconsistent images
    2. Remove repeated images.
"""

import glob
import os
import argparse
import subprocess
import json
from pprint import pprint

query_sets = ['flickr2k', 'flickr_tag', 'imagenet', 'imagenet_21k']
l_partitions = ['l0', 'lother']
file_types = {
    'json': ['json', 'keyword_*/*.json', '.json'],
    'orig': ['original', 'keyword_*/*.png', '.png'],
    'down': ['x4', 'keyword_*/*.png', '.png']
}


def parse_image_name(img_name):
    img_name_split = os.path.splitext(img_name)[0].split('/')
    return os.path.join(*img_name_split[-3:])


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Count number of images.')
    parser.add_argument('--save_dir', default='../flickr_test', type=str)
    # /cluster/work/cvl/videosr/flickr_data
    args = parser.parse_args()
    photo_id = dict()

    # remove inconsistent files
    for q_set in query_sets:
        for lp in l_partitions:
            # get all of the files in the three folders json, original, and x4.
            query_set_img_names = dict()
            for key, (folder, regex, fmt) in file_types.items():
                save_dir = os.path.join(args.save_dir, q_set, folder, lp, regex)
                img_names = set(map(parse_image_name, glob.glob(save_dir)))
                query_set_img_names[key] = [os.path.join(args.save_dir, q_set, folder), fmt, img_names]
            # pprint(query_set_img_names)

            # get the intersection of files
            intersec_img_names = query_set_img_names['json'][2].intersection(query_set_img_names['orig'][2],
                                                                             query_set_img_names['down'][2])
            # remove files
            for key, (root_dir, fmt, img_names) in query_set_img_names.items():
                diff_img_names = img_names - intersec_img_names
                pprint(diff_img_names)
                for name in diff_img_names:
                    img_name = os.path.join(root_dir, name + fmt)
                    subprocess.run(["rm", img_name])

    # remove repeated images
    for q_set in query_sets:
        for lp in l_partitions:
            json_dir = os.path.join(args.save_dir, q_set, 'json', lp, 'keyword_*/*.json')
            json_files = glob.glob(json_dir)
            for j_file in json_files:
                with open(j_file, 'r') as f:
                    info = json.load(f)
                hr_img_name = j_file.replace('/json/', '/original/').replace('.json', '.png')
                lr_img_name = j_file.replace('/json/', '/x4/').replace('.json', '.png')
                # print(j_file)
                # print(hr_img_name)
                # print(lr_img_name)
                # print(os.path.isfile(j_file), os.path.isfile(hr_img_name), os.path.isfile(lr_img_name))
                if info['photo_id'] not in photo_id:
                    photo_id[info['photo_id']] = 1
                else:
                    photo_id[info['photo_id']] += 1
                    print(j_file)
                    subprocess.run(["rm", j_file])
                    subprocess.run(["rm", hr_img_name])
                    subprocess.run(["rm", lr_img_name])
    pprint(photo_id)

