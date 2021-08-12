#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   data_prepare_normal.py
@Time    :   2021/08/10 14:15:26
@Author  :   Dio Chen
@Version :   1.0.0
@Contact :   zhuo.chen@wuerzburg-dynamics.com
@License :   None
@Desc    :   None
'''

# here put the import lib
import os
import json

import pandas as pd

from .data_prepare_3d import DataPrepare3D, get_area
from .data_utils import to_distribution_dict, to_dataset_dict_nothick


class DataPrepareNormal(DataPrepare3D):
    def __init__(self, config):
        super(DataPrepareNormal, self).__init__(config)

    def process_normal(self):
        if not os.path.exists(self.crop_dataset_path):
            os.mkdir(self.crop_dataset_path)

        data_list = pd.read_json(self.dataset_json)

        img_path = list(data_list['image'])
        mask_path = list(data_list['mask'])

        kidney_info = []
        for i in range(len(img_path)):

            img_p = img_path[i]
            mask_p = mask_path[i]

            # load data
            img_arr, mask_arr, case_name, ct_info, mask_info = self.load_data(img_p, mask_p)

            if len(mask_arr.shape) != 3 or len(
                    img_arr.shape) != 3 or img_arr.shape != mask_arr.shape or ct_info['space'] != mask_info['space']:
                print('error', img_arr.shape, mask_arr.shape, ct_info['space'], mask_info['space'])
                continue

            # preprocess
            img_arr, mask_arr, ct_info, old_spacing = self.preprocess(img_arr, mask_arr, ct_info)

            # get each kidney area
            kidney_area = self.get_kidney_area(mask_arr)

            for i, info in enumerate(kidney_area):
                save_name = case_name + '_' + str(i)

                # crop kidney area
                mask_crop, img_crop, is_normal, crop_edge = self.crop_area(mask_arr, img_arr, info)

                # save data
                self.save_data(img_crop, mask_crop, save_name, ct_info)

                # get kidney information dict
                kidney_dict = self.get_kidney_dict(save_name,
                                                   img_arr.shape,
                                                   old_spacing,
                                                   ct_info['space'],
                                                   area_info=info,
                                                   is_normal=is_normal,
                                                   crop_edge=crop_edge)

                # get renal medulla size information
                kidney_dict = self.get_stone_info(mask_crop.copy(), kidney_dict)

                kidney_info.append(kidney_dict)

        # save data detail to json
        self.info_to_json(self.data_detail_json, kidney_info)

    @staticmethod
    def get_stone_info(mask, info_dict):
        """ get stone size information. will copy the array first

        Args:
            mask (ndarray): mask array
            info_dict (dict): kidney info dict

        Returns:
            info_dict: kidney info dict with the size of stone
        """
        print('get_stone_info')
        mask_copy = mask.copy()

        mask_copy[mask_copy == 2] = 0

        stone_area = get_area(mask_copy)

        stone_num = len(stone_area)

        stone_areas = []

        if stone_area:
            for rm in stone_area:
                stone_areas.append(int(rm['voxel']))
                stone_size = sum(stone_areas)
        else:
            stone_areas = None
            stone_size = 0

        info_dict["stone_num"] = int(stone_num)
        info_dict["stone_areas"] = stone_areas
        info_dict["stone_size"] = int(stone_size)

        return info_dict

    def to_stone_dataset_json(self):
        """split dataset according to kidney info. and save to json
        """
        info_path = self.data_detail_json
        output_path = self.data_classify_json
        
        info_df = pd.read_json(info_path)
        info_df = info_df[info_df['normal_kidney']]
        analyzing = {
            '0': (info_df['stone_size'] == 0),
            '1_100': (info_df['stone_size'] > 0) & (info_df['stone_size'] < 100),
            '100_1000': (info_df['stone_size'] >= 100) & (info_df['stone_size'] < 1000),
            '1000_5000': (info_df['stone_size'] >= 1000) & (info_df['stone_size'] < 5000),
            '5000+': (info_df['stone_size'] >= 5000),
        }

        analyzing_dcit = to_distribution_dict(info_df, analyzing, group_by=None)

        dataset_dict = to_dataset_dict_nothick(analyzing_dcit, ['case_name', 'size', 'dataset'])

        print(dataset_dict)
        with open(output_path, 'w') as f:
            json.dump(dataset_dict, f)
