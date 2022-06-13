"""
Author: Yawei Li
Email: yawei.li@vision.ee.ethz.ch

Blur detector and flat region detector.
"""

import cv2
import os
import glob
import math
import numpy as np

def image_partition(image, grid_size):
    height, width = image.shape[:2]
    num_row, num_col = math.ceil(height / grid_size), math.ceil(width / grid_size)
    image_list = []
    for i in range(num_row):
        for j in range(num_col):
            image_list.append(
                image[i * grid_size: (i + 1) * grid_size, j * grid_size: (j + 1) * grid_size]
            )
    # print(f"Image shape ({height}, {width}), num_grid {len(image_list)} = {num_row} * {num_col}")
    return image_list


def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()


def variance_of_intensity(image):
    return image.var()


def variance_of_derivative(image):
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1)
    # abs_grad_x = cv2.convertScaleAbs(gx)
    # abs_grad_y = cv2.convertScaleAbs(gy)
    # grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    mag, ang = cv2.cartToPolar(gx, gy)
    return mag.var()


# def quality_metrics(image, blur_threshold, flat_threshold):
#     blur_score = variance_of_laplacian(image)
#     flat_score = variance_of_derivative(image)
#     blur_text = "Not Blurry" if blur_score >= blur_threshold else "Blurry"
#     flat_text = "Not Flat" if flat_score >= flat_threshold else "Flat"
#     return blur_score, flat_score, blur_text, flat_text
#
#
#
# def quality_metrics_grid(image, blur_threshold, flat_threshold, vote_threshold=0.4, grid_size=240):
#     blur_score = []
#     flat_score = []
#     blur_vote = []
#     flat_vote = []
#     image_list = image_partition(image, grid_size)
#     for image in image_list:
#         var_l = variance_of_laplacian(image)
#         var_i = variance_of_derivative(image)
#         blur_score.append(var_l)
#         flat_score.append(var_i)
#         blur_vote.append(var_l < blur_threshold)
#         flat_vote.append(var_i < flat_threshold)
#     # floats_array = np.array(flat_score)
#     # np.set_printoptions(precision=2)
#     # print(floats_array)
#     blur_score = sum(blur_score) / len(blur_score)
#     flat_score = sum(flat_score) / len(flat_score)
#     blur_vote = sum(blur_vote) / len(blur_vote)
#     flat_vote = sum(flat_vote) / len(flat_vote)
#     blur_text = "Not Blurry" if blur_vote < vote_threshold else "Blurry"
#     flat_text = "Not Flat" if flat_vote < vote_threshold else "Flat"
#
#     return blur_score, flat_score, blur_text, flat_text, blur_vote, flat_vote


def quality_metric(image, operator, threshold, grid=False, vote_threshold=0.4, grid_size=240):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # decide the operator
    if operator == "laplacian":
        func = variance_of_laplacian
        text = "Blurry"
    elif operator == "derivative":
        func = variance_of_derivative
        text = "Flat"
    elif operator == "intensity":
        func = variance_of_intensity
        text = "Flat"
    else:
        raise NotImplementedError(f"Operator {operator} not implemented.")

    # whether use grid mode or not
    if not grid:
        score = func(image)
        clean_flag = score >= threshold
        text = f"Not {text}" if clean_flag else text
        return clean_flag, score, text, -1
    else:
        # vote for blurry images.
        score, vote = [], []
        image_list = image_partition(image, grid_size)
        for image in image_list:
            var = func(image)
            score.append(var)
            vote.append(var < threshold)
        # floats_array = np.array(score)
        # np.set_printoptions(precision=2)
        # print(floats_array)
        score = sum(score) / len(score)
        vote = sum(vote) / len(vote)
        clean_flag = vote < vote_threshold
        text = f"Not {text}" if clean_flag else text
        # vote is between 1 and 0. The larger the vote, the more blurry the image.
        return clean_flag, score, text, vote


def sort_key_function(img_name):
    return int(os.path.splitext(os.path.basename(img_name))[0])


def save_image(image, save_dir, with_label=True, blur_text=0, blur_score=0, blur_vote=0, flat_text=0, flat_score=0, flat_vote=0):
    if with_label:
        min_dim = min(image.shape[:2])
        font_size = int(min_dim / 1000)
        y = int(min_dim / 1000) * 30

        blur_text = f"{blur_text}: {blur_score:.2f}, {blur_vote:.2f}"
        flat_text = f"{flat_text}: {flat_score:.2f}, {flat_vote:.2f}"

        cv2.putText(image, blur_text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 255), 3)
        cv2.putText(image, flat_text, (10, y * 2), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 255), 3)
    cv2.imwrite(save_dir, image)

if __name__ == "__main__":
    t_blur = 150
    t_flat = 150
    op_blur = "laplacian"
    op_flat = "derivative"
    grid = True
    t_vote = 0.5
    grid_size = 240

    # root_dir = "/Users/yaweili/projects/data_scrapy/flickr/flickr2k/l0/partition1"
    root_dir = "//flickr_final2_1000_per_user/flickr2k/lother/partition1"
    save_partition_name = "partition1_labelled_derivative"
    if grid:
        save_partition_name += "_grid"
    save_dir = root_dir.replace("partition1", save_partition_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    img_list = sorted(glob.glob(os.path.join(root_dir, "*.png")), key=sort_key_function)

    non_blur_counter, non_flat_counter, clean_counter = 0, 0, 0

    for i, imagePath in enumerate(img_list):
        img_name = os.path.splitext(os.path.basename(imagePath))[0]
        save_dir = imagePath.replace("partition1", save_partition_name)
        image = cv2.imread(imagePath)

        non_blur, blur_score, blur_text, blur_vote = quality_metric(image, op_blur, t_blur, grid=False,
                                                                    vote_threshold=t_vote, grid_size=grid_size)
        non_flat, flat_score, flat_text, flat_vote = quality_metric(image, op_flat, t_flat, grid=grid,
                                                                    vote_threshold=t_vote, grid_size=grid_size)
        save_image(image, save_dir, with_label=True, blur_text=blur_text, blur_score=blur_score, blur_vote=blur_vote,
                 flat_text=flat_text, flat_score=flat_score, flat_vote=flat_vote)

        print_text = f"Image {img_name:>4}: {blur_score:>8.2f} / {flat_score:>8.2f} |" \
                     f" {blur_vote:>2.2f} / {flat_vote:>2.2f} |" \
                     f" {blur_text:>15} / {flat_text:>15}"
        print(print_text)

        non_blur_counter += non_blur
        non_flat_counter += non_flat
        clean_counter += non_blur and non_flat

    print(f"Non blurry images {non_blur_counter}, non flat images {non_flat_counter}, clean images {clean_counter}")

    # cv2.imshow("Image", image)
    # key = cv2.waitKey(0)

    # gx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
    # gy = cv2.Sobel(image, cv2.CV_32F, 0, 1)
    # mag, ang = cv2.cartToPolar(gx, gy)
    # cv2.imshow("Image derivative", mag)
    # key = cv2.waitKey(0)

    # gx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
    # gy = cv2.Sobel(image, cv2.CV_32F, 0, 1)
    # abs_grad_x = cv2.convertScaleAbs(gx)
    # abs_grad_y = cv2.convertScaleAbs(gy)
    # grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    # cv2.imshow("Image derivative", grad)
    # # key = cv2.waitKey(0)
    # cv2.imwrite(imagePath.replace("partition1", save_partition_name), grad)