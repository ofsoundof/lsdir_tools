"""
Author: Yawei Li
Email: yawei.li@vision.ee.ethz.ch

Search and download image from flickr.com using Flickr API.
"""


import json
import os
import time
import concurrent.futures
import cv2
import numpy as np
import flickrapi
import copy
import argparse
import threading
from PIL import Image
from datetime import datetime
from urllib import request
import requests
from io import BytesIO
from functools import partial
from quality_evaluator import quality_metric, save_image
import glob
from pprint import pprint
thread_local = threading.local()
# requirements: opencv-python, flickrapi

def get_session():
    if not hasattr(thread_local, "session"):
        thread_local.session = requests.Session()
    return thread_local.session

# TODO: total number of images
class FlickrDownloader(object):
    def __init__(self, args):
        """
        self.query_img_info_list = {'photo_id1': info1, 'photo_id2': info2}
        self.query_img_list = {'photo_id1': photo1, 'photo_id2': photo2}


        Mapping from keyword id to keyword.
        self.cur_searched_keyword = {'1': {'keyword': keyword1, 'num': num1}, '2': {'keyword': keyword2, 'num': num2}}
            updated in search

        Information of photos in the current query set.
        The most import information is the photo id, which can be used to get other information such as camera info.
        self.cur_queryset_img_info = {}
            updated in save_image


        Information of searched users and photos.
        Used to filter photos according to the meta data returned by flickr.photos.search
        self.all_user_id = {'owner_id1': [date1, date2], 'owner_id2': [date1, date2, date3, date4]}
            updated in photo_filter
        self.all_photo_id = ['photo_id1', 'photo_id2', 'photo_id3']
            updated in photo_filter
        :param args:
        """
        self.imagenet_21k_dir = 'https://storage.googleapis.com/bit_models/imagenet21k_wordnet_lemmas.txt'
        self.imagenet_dir = 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
        # list of image information and downloaded images for each query
        self.query_img_info_list = {}
        self.query_img_list = {}
        self.query_img_save_list = []

        # state variable of the current query set.
        self.cur_searched_keyword = dict()     # updated in search
        self.cur_queryset_img_info = dict()    # updated in save_image

        # filtering criteria: list of searched users and photos
        self.all_user_id = dict()   # updated in photo_filter
        self.all_photo_id = dict()      # updated in photo_filter

        self.query_l0_num_img = 0      # -> self.cur_searched_keyword
        self.query_lother_num_img = 0
        self.query_set_num_img = 0
        # self.all_num_img_tmp

        self.blocked_tags = {'woman', 'women', 'man', 'men', 'lady', 'ladies', 'gentleman', 'gentlemen', 'girl', 'girls', 'boy',
                        'boys', 'guy', 'guys', 'female', 'male', 'people', 'sex', 'sexy', 'sensual', 'sensuality', 'boobs',
                        'breasts', 'tits', 'nude', 'naked', 'secondlife', 'children', 'kids', 'beauty'}
        self.args = args
        self.scale = args.scale
        self.start_page = args.start_page
        self.end_page = args.end_page
        self.api_key = args.api_key
        self.secret = args.secret
        self.query_sets = args.query_sets
        self.camera_info = args.camera_info
        self.flickr = flickrapi.FlickrAPI(self.api_key, self.secret, format='parsed-json')

        # save local information
        self.local_info_dir = os.path.join(self.args.save_dir, self.query_sets + '_info.json')
        # save global information
        self.global_info_dir = os.path.join(self.args.save_dir, "photo_user_list.json")
        # save query sets
        self.query_sets_dir = os.path.join(self.args.save_dir, "query_sets.json")
        # number of images
        self.num_img_dir = os.path.join(self.args.save_dir, "number_of_images.json")

        # Get query sets of the dateset: imagenet, imagenet_21k, flickr2k, flickr_tag
        self.all_sets = self.get_query_sets()
        self.licenses = self.get_license()
        self.search_kwargs = {
            'license': args.license,
            'sort': 'interestingness-desc',
            'is_gallery': args.in_gallery,
            'extras': args.extras,
            'per_page': args.per_page,
            'content_type': 1,
            'min_upload_date': datetime.timestamp(datetime.strptime(args.start_date, '%m/%d/%Y'))
        }

    def get_imagenet_label(self, online_dir):
        label_file = request.urlopen(online_dir)
        labels = []
        suffix = ['boy', 'girl', 'woman', 'man', 'lady', 'father', 'mother', 'person', 'guy', 'ese']
        for line in label_file:
            decoded_line = line.decode('utf-8').replace('\n', '').replace(' ', '').split(',')
            flag = True
            if len(decoded_line) == 1 and decoded_line[0] in ['sensualist', 'bikini', 'people']:
                flag = False
            for s1 in decoded_line:
                for s2 in suffix:
                    if s1.endswith(s2):
                        flag = False
            if flag:
                labels.append(decoded_line[0])
        return labels

    def get_flickr_tag(self):

        # get the most popular tags during the last week
        print('Get the most popular tags.')
        hot_tags = self.flickr.tags.getHotList(api_key=self.api_key, period='week', count=200)
        hot_tags = set(h['_content'] for h in hot_tags['hottags']['tag'])
        all_hot_tags = copy.deepcopy(hot_tags)
        # print(all_hot_tags)

        print('Get related tags.')
        for h_tag in hot_tags:
            related_tags = self.flickr.tags.getRelated(api_key=self.api_key, tag=h_tag)
            related_tags = [h['_content'] for h in related_tags['tags']['tag']]
            all_hot_tags.update(related_tags)
            # print(related_tags)
            # print(h_tag, related_tags)
            # photos = flickr.photos.search(per_page=500, page=1, content_type=1, tags=h_tag)
            # photos = photos['photos']['photo"]
            # print(f"Number of searched images {len(photos)}")
            # if not len(related_tags):
            #     from IPython import embed; embed(); exit()
        all_hot_tags = all_hot_tags - self.blocked_tags
        print(f"Number of found Flickr tags: {len(all_hot_tags)}")
        print(all_hot_tags)

        return list(all_hot_tags)

    @staticmethod
    def flickr2k_query():
        # queries = [['animal']]
        queries = ['animal', 'plants', 'city', 'nature']
        queries.extend([
            'fish', 'mammal', 'bird', 'insect',
            'fruits', 'flower', 'tree',
            'street', 'cityscape', 'building', 'market',
            'seaside', 'lake', 'valley', 'forest',
            'food', 'crowd'])
        queries.extend([
            'mushroom', 'coral', 'rose', 'orange', 'onion', 'leaf',
            'lion', 'tiger', 'horse', 'cat', 'monkey', 'peacock', 'fox', 'bear', 'squirrel', 'bee', 'starfish', 'spider',
            'apartment', 'architecture', 'train', 'car', 'university', 'garden', 'park',
            'coast', 'beach', 'climbing', 'mountain', 'garden', 'snow', 'bread',
            'photographic', 'hawai', 'dubai', 'vegas', 'seoul'])
        queries.extend([
            'jungle', 'landscape', 'spring', 'summer', 'autumn', 'winter', 'view', 'sightseeing', 'scenery',
            'hotel', 'bar', 'library', 'photographs', 'audience', 'interior', 'army', 'tribes', 'junk', 'trash',
            'pabble', 'centralpark', 'machinery', 'semiconductor', 'zoo', 'farm', 'hockey', 'tourdefrance', 'festival',
            'museum', 'poverty', 'war', 'microscopic', 'structure', 'orchestra', 'traditional'])
        queries.extend([
            'school', 'blossom', 'cherry', 'Reykjavik', 'milkyway', 'restaurant', 'penguin', 'rabbit', 'italy', 'cow',
            'wallpaper', 'strawberry', 'boar', 'temple', 'sculpture', 'beauty', 'complicated', 'bush', 'themepark',
            'parrot',
            'tourist', 'swiss', 'korea', 'india', 'tokyo', 'assam', 'dog', 'kebob', 'salad', 'fruitstore',
            'wheat', 'newyork', 'corn', 'kingcrab', 'graphity', 'timber', 'seashore', 'cremona', 'playground',
            'queenstown'])
        return queries

    def get_query_sets(self):
        if os.path.isfile(self.query_sets_dir):
            with open(self.query_sets_dir, 'r') as f:
                sets = json.load(f)
        else:
            flickr2k = set(self.flickr2k_query())
            imagenet = set(self.get_imagenet_label(self.imagenet_dir)) - flickr2k
            imagenet_21k = set(self.get_imagenet_label(self.imagenet_21k_dir)) - flickr2k.union(imagenet)
            flickr_tag = self.get_flickr_tag()
            flickr2k = list(flickr2k)
            imagenet = list(imagenet)
            imagenet_21k = list(imagenet_21k)
            sets = {'flickr2k': flickr2k, 'imagenet': imagenet, 'imagenet_21k': imagenet_21k, 'flickr_tag': flickr_tag}
            with open(self.query_sets_dir, 'w') as f:
                json.dump(sets, f)

        return_sets = dict()
        if self.query_sets == 'imagenet':
            return_sets['imagenet'] = sets['imagenet']
        if self.query_sets == 'imagenet_21k' or self.query_sets == 'all':
            return_sets['imagenet_21k'] = sets['imagenet_21k']
        if self.query_sets == 'flickr2k' or self.query_sets == 'all':
            return_sets['flickr2k'] = sets['flickr2k']
        if self.query_sets == 'flickr_tag' or self.query_sets == 'all':
            return_sets['flickr_tag'] = sets['flickr_tag']

        return return_sets

    def get_license(self):
        licenses = self.flickr.photos.licenses.getInfo(api_key=self.api_key, format='parsed-json')
        out = dict()
        for l in licenses['licenses']['license']:
            out[l['id']] = l['name']
        return out

    def filter_tags(self, photo):
        tags = set(photo['tags'].split(' '))
        intersection = self.blocked_tags & tags
        return len(intersection) > 0 or len(tags) <= 1

    def get_buddyicon(self, photo):
        owner = photo['owner']
        icon_server = photo['iconserver']
        icon_farm = photo['iconfarm']
        if int(icon_server) > 0:
            buddyicon_url = f"http://farm{icon_farm}.staticflickr.com/{icon_server}/buddyicons/{owner}.jpg"
        else:
            buddyicon_url = "https://www.flickr.com/images/buddyicon.gif"
        return buddyicon_url

    def parse_camera_info(self, photo_id):
        all_camera_info = self.flickr.photos.getExif(api_key=self.api_key, photo_id=photo_id)
        camera_info = {'camera': all_camera_info['photo']['camera']}
        for camera_label in all_camera_info['photo']['exif']:
            if 'clean' in camera_label:
                camera_info[camera_label['label']] = camera_label['clean']['_content']
            elif camera_label['label'] in ['Lens Model', 'White Balance']:
                camera_info[camera_label['label']] = camera_label['raw']['_content']
        return camera_info

    def get_camera_info(self, photo):
        if self.camera_info:
            try:
                return self.parse_camera_info(photo['id'])
            except:
                # print('Camera information is not available')
                pass

    def append(self, photo):
        if photo['owner'] in self.all_user_id:
            self.all_user_id[photo['owner']].append(photo['datetaken'][:10])
        else:
            self.all_user_id[photo['owner']] = [photo['datetaken'][:10]]
        self.all_photo_id[photo['id']] = photo['url_o']
        self.query_img_info_list[photo['id']] = photo

    def photo_filter(self, photos, string):
        self.query_img_info_list = {}
        min_views = self.args.min_views
        min_faves = self.args.min_faves
        max_aspect_ratio = self.args.aspect_ratio
        resolution = self.args.resolution
        per_user = self.args.per_user

        for i, photo in enumerate(photos):
            views = int(photo['views'])
            photo_id = photo['id']
            owner = photo['owner']
            height = photo.get('height_o', 1)
            width = photo.get('width_o', 1)
            min_dim, max_dim = min(height, width), max(height, width)
            aspect_ratio = max_dim / min_dim
            print_str = string + f"[{i + 1}/{len(photos)}]: "

            keep_flag = True
            if not ('height_o' in photo and 'width_o' in photo and 'url_o' in photo):
                keep_flag = False
                print_str += f"Original URL does exists; "
            if views < min_views:
                keep_flag = False
                print_str += f"Views {views} less than {min_views}; "
            if min_dim < resolution:
                keep_flag = False
                print_str += f"Resolution {min_dim} lower than {resolution}; "
            if aspect_ratio > max_aspect_ratio:
                keep_flag = False
                print_str += f"Aspect ratio {aspect_ratio} larger than {max_aspect_ratio}; "
            if photo_id in self.all_photo_id:
                keep_flag = False
                print_str += f"Repeated image; "
            if len(self.all_user_id.get(owner, [])) >= per_user:
                keep_flag = False
                print_str += f"Only {per_user} images is allowed for each user; "
            if photo["datetaken"][:10] in self.all_user_id.get(owner, []):
                keep_flag = False
                print_str += f"Images taken by the same user on the same date is not allowed; "
            if self.filter_tags(photo):
                keep_flag = False
                print_str += f"Blocked tags occurs in the current photo tags;"
            print(print_str)

            if keep_flag:
                if min_faves > 0:
                    fave_info = self.flickr.photos.getFavorites(api_key=self.api_key, photo_id=photo_id)
                    faves = int(fave_info['photo']['total'])
                    if faves >= min_faves:
                        self.append(photo)
                    else:
                        print(print_str + f"Faves {faves} less than {min_faves}")
                else:
                    self.append(photo)

    def _img_quality_check(self, img_id):
        img = self.query_img_list[img_id]
        img_info = self.query_img_info_list[img_id]

        # res = cv2.GaussianBlur(img, (self.args.smooth_kernel, self.args.smooth_kernel), 0)
        # res = cv2.resize(res, None, fx=1 / self.args.scale, fy=1 / self.args.scale, interpolation=cv2.INTER_LANCZOS4)

        # PIL.Image.resize leads to the same results with Matlab
        width, height = img_info['width_o'], img_info['height_o']
        res = np.asarray(img.resize((width // self.scale, height // self.scale), resample=Image.LANCZOS))
        res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)

        # blur detection
        non_blur, blur_score, blur_text, blur_vote = quality_metric(res, 'laplacian', self.args.t_blur_low, grid=False)
        non_blur = non_blur and blur_score <= self.args.t_blur_up  # Large blur score usually means noisy images.

        # flat region detection
        if non_blur & self.args.flat_detector:
            non_flat, flat_score, flat_text, flat_vote = quality_metric(
                res, 'derivative', self.args.t_flat, grid=True, vote_threshold=self.args.t_vote,
                grid_size=self.args.grid_size
            )
        else:
            non_flat, flat_score, flat_text, flat_vote = True, -1, 'Flat detector not available', -1
        # print(img_id, non_flat, non_blur)
        if non_flat and non_blur:
            height = 48 * int(height / 48)
            width = 48 * int(width / 48)
            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            img = img[:height, :width, :]
            # min_dim = min(img.shape[:2])
            # font_size = int(min_dim / 1000)
            # y = int(min_dim / 1000) * 30
            # blur_text = f"{blur_text}: {blur_score:.2f}, {blur_vote:.2f}"
            # flat_text = f"{flat_text}: {flat_score:.2f}, {flat_vote:.2f}"
            # view_text = f"#Views {img_info['views']}"
            # keyword_text = f"Keyword: {query}"
            # cv2.putText(img, keyword_text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 255), 3)
            # cv2.putText(img, blur_text, (10, y * 2), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 255), 3)
            # cv2.putText(img, flat_text, (10, y * 3), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 255), 3)
            # cv2.putText(img, view_text, (10, y * 4), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 255), 3)
            # cv2.putText(img, img_info['tags'], (10, y * 5), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 255), 3)
            self.query_img_save_list.append([img, res, img_info, blur_score, flat_vote])
            # print(len(self.query_img_save_list))

    def quality_filter(self):
        self.query_img_save_list = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            executor.map(self._img_quality_check, self.query_img_info_list.keys())

        for img_save_list in self.query_img_save_list:
            if img_save_list[2]['license'] == '0':
                self.query_l0_num_img += 1
                img_save_list.insert(0, self.query_l0_num_img)
            else:
                self.query_lother_num_img += 1
                img_save_list.insert(0, self.query_lother_num_img)

    def _get_out_info(self, photo_info, query, blur, flat):
        output_photo_info = {
            'keyword': query,
            'photo_id': photo_info['id'],
            'date_upload': int(photo_info['dateupload']),
            'date_taken': datetime.timestamp(datetime.strptime(photo_info['datetaken'], '%Y-%m-%d %H:%M:%S')),
            'license': self.licenses[photo_info['license']],
            'height': photo_info['height_o'],
            'width': photo_info['width_o'],
            'url': photo_info['url_o'],
            'blur_score': blur,
            'flat_vote_score': flat,
        }
        for attr in ['owner', 'ownername', 'title', 'latitude', 'longitude', 'accuracy', 'tags']:
            output_photo_info[attr] = photo_info[attr]
        output_photo_info.update({'buddyicon_url': self.get_buddyicon(photo_info),
                                  'camera': self.get_camera_info(photo_info)})
        return output_photo_info

    def _prepare_save_dir(self, img_info, query_set, query, query_id, counter):
        # license partition
        l_partition = 'l0' if img_info['license'] == '0' else 'lother'
        # keyword partition
        partition = f"keyword_{query_id + 1}"
        # get save directory
        root_dir = os.path.join(self.args.save_dir, query_set)
        save_dirs = []
        for d in ['original', 'x4', 'json']:
            save_dir = os.path.join(root_dir, d, l_partition, partition)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_dirs.append(save_dir)
        save_dir_summary = os.path.join(save_dirs[0], f"{query}.txt")
        save_dir_original = os.path.join(save_dirs[0], f"{counter}.png")
        save_dir_downscale = os.path.join(save_dirs[1], f"{counter}.png")
        save_dir_json = os.path.join(save_dirs[2], f"{counter}.json")

        return l_partition, partition, save_dir_summary, save_dir_original, save_dir_downscale, save_dir_json

    def save_image(self, inputs, query_set, query, query_id, string):
        counter, img, res, img_info, blur, flat = inputs
        out_img_info = self._get_out_info(img_info, query, blur, flat)

        try:
            # prepare save dir
            l_partition, partition, dir_summ, dir_orig, dir_down, dir_json \
                = self._prepare_save_dir(img_info, query_set, query, query_id, counter)
            # update dataset_info
            self.cur_queryset_img_info[f"{query_set}_{l_partition}_{partition}_{counter}"] = out_img_info

            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
            # Image.fromarray(img).save(dir_orig)
            # Image.fromarray(res).save(dir_down)
            cv2.imwrite(dir_orig, img)
            cv2.imwrite(dir_down, res)
            with open(dir_json, 'w') as f:
                json.dump(out_img_info, f)
            with open(dir_summ, 'a') as f:
                s = f"{counter:8d}\t{img_info['id']}\t{img_info['url_o']}\n"
                f.write(s)

            print(string)
            # print('current number of images', self.query_l0_num_img + self.query_lother_num_img)

        except:
            print('\t-> Error: writing image')

    def save_images(self, query_set, query, query_id, string):
        num_save_imgs = len(self.query_img_save_list)
        if self.args.parallel_save:
            threads = []
            for j, save_info in enumerate(self.query_img_save_list):
                print_str = string + f"[{j + 1}/{num_save_imgs}]: {save_info[-3]['title'][:50]}"
                t = threading.Thread(target=self.save_image,
                                     args=(save_info, query_set, query, query_id, print_str))
                t.start()
                threads.append(t)

            for t in threads:
                t.join()
            # save_image_fun = partial(self.save_image, query_set=query_set, query=query, query_id=query_id, string=string)
            # with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            #     executor.map(save_image_fun, img_save_list)
        else:
            for j, save_info in enumerate(self.query_img_save_list):
                print_str = string + f"[{j + 1}/{num_save_imgs}]: {save_info[-3]['title'][:50]}"
                self.save_image(save_info, query_set=query_set, query=query, query_id=query_id, string=print_str)

    def download_image(self, photo):
        session = get_session()
        with session.get(photo['url_o']) as response:
            image = Image.open(BytesIO(response.content)).convert('RGB')
            print(f"Download from {photo['url_o']}")
            self.query_img_list[photo['id']] = image

    def download_all_images(self):
        self.query_img_list = {}
        # print('Start download')
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(self.download_image, self.query_img_info_list.values())

    def update_checkpoint(self, query_set, query, query_id):
        # update cur_searched_keyword, cur_queryset_img_info
        self.cur_searched_keyword[str(query_id + 1)] = \
            {'keyword': query, 'num_l0': self.query_l0_num_img, 'num_lother': self.query_lother_num_img}
        try:
            if os.path.isfile(self.local_info_dir):
                with open(self.local_info_dir, 'r') as f:
                    info = json.load(f)
                info['cur_searched_keyword'].update(self.cur_searched_keyword)
                info['cur_queryset_img_info'].update(self.cur_queryset_img_info)
            else:
                info = {'cur_searched_keyword': self.cur_searched_keyword,
                        'cur_queryset_img_info': self.cur_queryset_img_info}
            self.cur_searched_keyword = info['cur_searched_keyword']
            self.cur_queryset_img_info = info['cur_queryset_img_info']

            with open(self.local_info_dir, 'w') as f:
                json.dump(info, f)
        except:
            print(f"Updating current information failed.")

        # update query_set_num_img
        self.query_set_num_img += self.query_l0_num_img + self.query_lother_num_img
        try:
            if os.path.isfile(self.num_img_dir):
                with open(self.num_img_dir, 'r') as f:
                    all_num_img = json.load(f)
                all_num_img[query_set] = all_num_img.get(query_set, 0) + self.query_set_num_img
            else:
                all_num_img = {query_set: self.query_set_num_img}

            with open(self.num_img_dir, 'w') as f:
                json.dump(all_num_img, f)
            self.query_set_num_img = 0
        except:
            print(f"Updating number of images failed.")

        try:
            # update all_photo_id and all_user_id
            if os.path.isfile(self.global_info_dir):
                with open(self.global_info_dir, 'r') as f:
                    info = json.load(f)
                info['all_photo_id'].update(self.all_photo_id)
                info['all_user_id'].update(self.all_user_id)
            else:
                info = {'all_photo_id': self.all_photo_id,
                        'all_user_id': self.all_user_id}
            self.all_photo_id = info['all_photo_id']
            self.all_user_id = info['all_user_id']

            with open(self.global_info_dir, 'w') as f:
                json.dump(info, f)
        except:
            print(f"Updating photo_id and user_id failed.")

    def _init_for_query(self, query_id):
        # if os.path.isfile(self.local_info_dir):
        #     try:
        #         with open(self.local_info_dir, 'r') as f:
        #             pre_searched_keyword = json.load(f)['cur_searched_keyword']
        #
        #         if str(query_id + 1) not in pre_searched_keyword:
        #             self.query_l0_num_img = 0
        #             self.query_lother_num_img = 0
        #         else:
        #             self.query_l0_num_img = pre_searched_keyword[str(query_id + 1)]['num_l0']
        #             self.query_lother_num_img = pre_searched_keyword[str(query_id + 1)]['num_lother']
        #     except:
        #         self.query_l0_num_img = 0
        #         self.query_lother_num_img = 0
        # else:
        #     self.query_l0_num_img = 0
        #     self.query_lother_num_img = 0

        self.query_l0_num_img = 0
        self.query_lother_num_img = 0
        if os.path.isfile(self.local_info_dir):
            try:
                with open(self.local_info_dir, 'r') as f:
                    pre_searched_keyword = json.load(f)['cur_searched_keyword']
                if str(query_id + 1) in pre_searched_keyword:
                    self.query_l0_num_img = pre_searched_keyword[str(query_id + 1)]['num_l0']
                    self.query_lother_num_img = pre_searched_keyword[str(query_id + 1)]['num_lother']
            except:
                print('Update number of images per query failed.')


    def search(self):
        # Iterate the query sets
        for key, query_set in self.all_sets.items():
            print(f"Start searching query set {key} ==>")
            for i, query in enumerate(query_set):
                if self.args.start_id <= i + 1 <= self.args.end_id:
                    self._init_for_query(i)
                    print(f"Start searching keyword {key}:{query} ==>")
                    for page in range(self.start_page, self.end_page + 1):
                        # search images
                        try:
                            if key == 'flickr_tag':
                                photos = self.flickr.photos.search(tags=query, page=page, **self.search_kwargs)
                            else:
                                photos = self.flickr.photos.search(text=query, page=page, **self.search_kwargs)
                        except:
                            continue
                        # filter images according to the meta data
                        string = f"[Filtering {key}][{i + 1}/{len(query_set)}]" \
                                 f"(query: {query})[{page}/{self.end_page}]"
                        self.photo_filter(photos['photos']['photo'], string)
                        # download images
                        self.download_all_images()
                        # filter images according to blur and flat detection
                        self.quality_filter()
                        # write images
                        string = string.replace('Filtering', 'Writing')
                        self.save_images(query_set=key, query=query, query_id=i, string=string)
                    # update checkpoint
                    self.update_checkpoint(query_set=key, query=query, query_id=i)
                    print(f"Finish searching keyword {key}: "
                          f"({query}, {self.query_l0_num_img + self.query_lother_num_img}, "
                          f"{self.query_l0_num_img}, {self.query_lother_num_img}) ==>")
            print(f"Finish searching query set {key} ==>")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Collecting high quality images from Flickr.')
    parser.add_argument('--api_key', default='', type=str, help='You need to set your own api key.')
    parser.add_argument('--secret', default='', type=str, help='You need to get your own secret.')
    parser.add_argument('--save_dir', default='/cluster/work/cvl/videosr/flickr_data', type=str)
    # /cluster/work/cvl/videosr
    parser.add_argument('--extras',
                        default='license, date_upload, date_taken, owner_name, icon_server, geo, tags, views, url_o',
                        help='Get extra information of the search images.')
    parser.add_argument('--license', default='1,2,3,4,5,6,7,8,9,10', type=str,
                        help='License of the searched images')
    parser.add_argument('--in_gallery', action='store_true',
                        help='Only search images in a gallery')
    parser.add_argument('--start_date', default='01/01/2017', type=str,
                        help='Start date to search the photos')

    parser.add_argument('--t_blur_low', default=150., type=float, help='600')
    parser.add_argument('--t_blur_up', default=8000., type=float, help='8000')
    parser.add_argument('--t_flat', default=800., type=float, help='800')
    parser.add_argument('--t_vote', default=.5, type=float, help='0.2')
    parser.add_argument('--grid_size', default=240, type=int)
    parser.add_argument('--flat_detector', action='store_false')
    parser.add_argument('--scale', default=4, type=int)
    parser.add_argument('--start_id', default=1, type=int, help='Start ID of the query set.')
    parser.add_argument('--end_id', default=1000000, type=int, help='End ID of the query set.')
    parser.add_argument('--parallel_save', action='store_true')

    parser.add_argument('--camera_info', action='store_true', dest='camera_info')
    parser.add_argument('--per_page', default=400, type=int)
    parser.add_argument('--start_page', default=1, type=int)
    parser.add_argument('--end_page', default=1, type=int)
    parser.add_argument('--resolution', default=2160, type=int)
    parser.add_argument('--aspect_ratio', default=2.0, type=float)
    parser.add_argument('--min_views', default=0, type=int)
    parser.add_argument('--min_faves', default=0, type=int)
    parser.add_argument('--per_user', default=50, type=int, help='Maximum number of photos per user.')
    parser.add_argument('--query_sets', default='flickr2k', type=str,
                        choices=['imagenet', 'imagenet_21k', 'flickr2k', 'flickr_tag', 'all'])

    args = parser.parse_args()
    pprint(args)

    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)

    downloader = FlickrDownloader(args)
    downloader.search()
    print("Finished data crawling!")