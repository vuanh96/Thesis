import os
import json
import pandas as pd
import numpy as np
import argparse
from pascal_voc_writer import Writer
import shutil
from data import my_utils
import random


def parse_args():
    str_help = "\n".join(["Id : {} - Class Name : {}".format(k, v) for k, v in id2name.items()])

    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', required=True, help='dir contain mot data')
    parser.add_argument('--dst_dir', default="./voc_format_images", help='dir contain voc format data')
    parser.add_argument('--sel_object_ids', default="1,2,7", help='list contain selected object class ids\n{}'.format(str_help))
    parser.add_argument('--size', default="100,0", help='dataset size')
    # parser.add_argument('--valid_video_ids', default="02", help='video ids of valid set, example: 02')
    # parser.add_argument('--test_video_ids', default="11", help='video ids of test set, example: 11')
    parser.add_argument('--min_vis_ratio', default=0.05, type=float, help='min of visibility ratio')

    args = parser.parse_args()

    if args.sel_object_ids is None:
        args.sel_object_ids = []
    else:
        args.sel_object_ids = [int(id) for id in args.sel_object_ids.split(",")]

    # if args.valid_video_ids is None:
    #     args.valid_video_ids = []
    # else:
    #     args.valid_video_ids = ["MOT17-{}".format(id) for id in args.valid_video_ids.split(",") if len(id) > 0]
    #
    # if args.test_video_ids is None:
    #     args.test_video_ids = []
    # else:
    #     args.test_video_ids = ["MOT17-{}".format(id) for id in args.test_video_ids.split(",") if len(id) > 0]

    return args


# Mapping object class id to object name corresponding to mot format
id2name = {
    1: "human",
    2: "human",
    3: "Car",
    4: "Bicycle",
    5: "Motorbike",
    6: "Non motorized vehicle",
    7: "human",
    8: "artificial",
    9: "occluder",
    10: "occluder",
    11: "occluder",
    12: "artificial",
}

# id2name = {
#     1: "Pedestrian",
#     2: "Person on vehicle",
#     3: "Car",
#     4: "Bicycle",
#     5: "Motorbike",
#     6: "Non motorized vehicle",
#     7: "Static person",
#     8: "Distractor",
#     9: "Occluder",
#     10: "Occluder on the ground",
#     11: "Occluder full",
#     12: "Reflection",
# }

if __name__ == "__main__":
    args = parse_args()
    src_dir = args.src_dir
    dst_dir = args.dst_dir
    min_vis_ratio = args.min_vis_ratio
    dataset_size = args.size
    train_ratio, test_ratio = [int(s) for s in dataset_size.split(",")]

    mot_dirs = my_utils.get_dir_names(src_dir)
    # mot_dirs = ["MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09", "MOT17-10", "MOT17-11", "MOT17-13"]
    info_maps = {}
    gt_df_map = {}
    full_train_mot_paths = []
    for mot_dir_id, mot_dir in enumerate(mot_dirs):
        # Read seqinfo.ini file
        seq_info_path = os.path.join(src_dir, mot_dir, "seqinfo.ini")
        seq_info = my_utils.load_list(seq_info_path)
        info_map = {}
        for line in seq_info:
            lst = line.split("=")
            info_map.update({lst[0]: lst[-1]})
        # print(info_map)
        info_maps[mot_dir] = info_map

        # Read ground truth file
        gt_path = os.path.join(src_dir, mot_dir, "gt/gt.txt")
        col_names = ["FrameId", "ObjectId", "Left", "Top", "Width", "Height", "IgnoreFlag", "ObjectClass",
                     "VisibilityRatio"]
        gt_df = my_utils.load_csv(gt_path, names=col_names)
        gt_df_map[mot_dir] = gt_df
        # print(gt_df.head())
        # print("Number line ground truth : ", gt_df.shape[0])

        src_img_dir = os.path.join(src_dir, mot_dir, info_map["imDir"])
        src_img_paths = my_utils.get_file_paths(src_img_dir)
        full_train_mot_paths.extend(src_img_paths)

    # random.shuffle(full_train_mot_paths)
    len_dataset = len(full_train_mot_paths)
    print("Length dataset: %d" % len_dataset)
    if train_ratio > 0:
        train_mot_paths = full_train_mot_paths[:int(train_ratio/100*len_dataset)]
    else:
        train_mot_paths = []

    if test_ratio > 0:
        test_mot_paths = full_train_mot_paths[-int(test_ratio/100*len_dataset):]
    else:
        test_mot_paths = []

    for src_img_paths, sub_dir_name in [(train_mot_paths, "train")]:

        # make directory storage dataset
        sub_dst_dir = os.path.join(dst_dir, sub_dir_name)
        if os.path.exists(sub_dst_dir):
            shutil.rmtree(sub_dst_dir)
        os.mkdir(sub_dst_dir)
        os.mkdir(os.path.join(sub_dst_dir, "Annotations"))
        os.mkdir(os.path.join(sub_dst_dir, "JPEGImages"))
        os.mkdir(os.path.join(sub_dst_dir, "ImageSets"))
        os.mkdir(os.path.join(sub_dst_dir, "ImageSets", "Main"))

        img_list = []
        gt_json = {}

        for img_id, src_img_path in enumerate(src_img_paths):
            if (img_id + 1) % 100 == 0:
                print("{}/{} - Processing image ...".format(img_id + 1, len(src_img_paths)))

            img_name = os.path.basename(src_img_path)
            mot_dir = my_utils.get_dir_name_of_path(os.path.dirname(src_img_path))
            dst_img_name = "{}_{}".format(mot_dir, img_name)
            dst_img_path = os.path.join(sub_dst_dir, "JPEGImages", dst_img_name)

            xml_name = dst_img_name[:dst_img_name.rfind(".")] + ".xml"

            # Generate voc annotation file
            writer = Writer(dst_img_path, info_maps[mot_dir]["imWidth"], info_maps[mot_dir]["imHeight"])

            frame_id = int(img_name[:img_name.rfind(".")])
            gt_df = gt_df_map[mot_dir]
            frame_df = gt_df[gt_df["FrameId"] == frame_id]
            # print(frame_df.head())
            select_img = False
            gt_bboxes = []
            for df_idx, row in frame_df.iterrows():
                object_class_id, xmin, ymin, width, height, vis_ratio = row["ObjectClass"], row["Left"], \
                                                             row["Top"], row["Width"], row["Height"], row["VisibilityRatio"]
                if vis_ratio < min_vis_ratio:
                    continue

                if len(args.sel_object_ids) > 0 and object_class_id not in args.sel_object_ids:
                    continue
                select_img = True
                object_class_name = id2name[object_class_id]

                gt_bboxes.append({
                    "type": "gt",
                    "class_id": object_class_name,
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmin + width - 1,
                    "ymax": ymin + height - 1,
                })

                # Add Object
                xmax, ymax = xmin + width - 1, ymin + height - 1
                writer.addObject(object_class_name, int(xmin), int(ymin), int(xmax), int(ymax))

            # save voc annotation file
            if select_img:
                gt_json[dst_img_name] = gt_bboxes
                img_list.append("{}".format(dst_img_name[:dst_img_name.rfind('.')]))
                my_utils.copy_file(src_img_path, dst_img_path)
                xml_path = os.path.join(sub_dst_dir, "Annotations", xml_name)
                my_utils.make_parent_dirs(xml_path)
                writer.save(xml_path)

        # Generate img list
        img_list_path = os.path.join(sub_dst_dir, "ImageSets", "Main", "trainval.txt")
        my_utils.save_list(img_list, img_list_path)

        gt_save_path = os.path.join(sub_dst_dir, "gt.json")
        my_utils.save_json(gt_json, gt_save_path)
