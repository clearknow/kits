"""
    training config
    author: ChuDa xiao
"""
import torch
import os


class BaseConfig:
    def __init__(self):
        self.image_root = "/public/home/cxiao/Study/git_pro/LIDC-IDRI-Preprocessing/data/image/"
        self.mask_root = "/public/home/cxiao/Study/git_pro/LIDC-IDRI-Preprocessing/data/mask/"
        self.check_point = "checkpoints/"
        self.batch_size = 8
        self.epochs = 10
        self.lr = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")
        # the percentage of validation set
        self.val_percent = 0.1
        self.num_workers = 4
        # 0 denote crossEntropy; 1 denote dice Loss
        self.loss = 0
        self.img_scale = 0.5
        self.save_path = "/public/datasets/lung/predict/"
        # load model file path
        self.pre_model_path = ""
        self.data_parallel = False
        # os.environ['CUDA_VISIBLE_DEVICES'] = []
        self.device_ids = [2, 3]

        # resample CT image shape ()
        self.new_size = (128, 256, 256)


class Config(BaseConfig):
    def __init__(self):
        super(Config, self).__init__()
        '''
            set your config 
        '''
        self.batch_size = 4
        self.epochs = 40
        self.spacing = (3.22, 1.62, 1.62)
        self.entropy_weight = [1, 4, 8, 8]
        # dice loss
        self.loss = 0
        # artery image
        self.root_path = "/public/home/cxiao/Study/data/kits21"
        self.artery_image_root = "/public/home/cxiao/Study/data/artery/image"
        self.artery_mask_root = "/public/home/cxiao/Study/data/artery/mask/"
        self.artery_save_path = "/public/home/cxiao/Study/data/artery/predict/"
        self.second_network = False
        # kits
        self.kits_image_root = os.path.join(self.root_path, "image_2d")
        self.kits_mask_root = os.path.join(self.root_path, "mask_2d")
        self.kits_save_root = os.path.join(self.root_path, "predict")
        self.origin_data_path = os.path.join(self.root_path, "data")
        self.image_3d_path = os.path.join(self.root_path, "image")
        self.mask_3d_path = os.path.join(self.root_path, "mask")

        self.mask_3D_cube = os.path.join(self.root_path, "mask_cube")
        self.image_3D_cube = os.path.join(self.root_path, "image_cube")
        self.kits_val_image_path = os.path.join(self.root_path, "val/image")
        self.kits_val_mask_path = os.path.join(self.root_path, "val/mask")

        self.image_crop = os.path.join(self.root_path, "image_crop")
        self.mask_crop = os.path.join(self.root_path, "mask_crop")
        self.val_image_crop = os.path.join(self.root_path, "val/image_crop")
        self.val_mask_crop = "/public/home/cxiao/Study/data/kits21/val/mask_crop"

        self.image_json = "/public/home/cxiao/Study/data/kits21/val/images.json"

        self.image_root = self.kits_image_root
        self.mask_root = self.kits_mask_root

        self.val_image_path = self.kits_val_image_path
        self.val_mask_path = self.kits_val_mask_path
        self.save_path = self.kits_save_root
        self.network = "UNet"
        self.scheduler = ""
        self.optimizer = "Adam"
        self.pro_pre = True
        # load model file path
        self.weight = None
        self.first_model = ""
        self.second_model = ""
        self.root_path = "/public/home/cxiao/Study/data/kits21"

        self.dataset_json = ""
        self.layer_thick = 32

        self.thickness = 2
        self.ct_stage = "normal"  # normal, enh_a, enh_d

        self.area_size = (self.layer_thick, 384, 384)
        self.resize_xy = (512, 512)

        self.case_imgs_name = ".nii.gz"
        self.case_mask_name = ".nii.gz"
        self.windowing = [-79, 304]
        self.stride = 8
        self.trans_types = ['AffineNoTranslate', 'AffineNoTranslate', 'HorizontalFlip',
                            'HorizontalFlip_Affine', 'HorizontalFlip_Affine']
        self.change_label = {'2': 0, '3': 0}

        # crop origin dataset shape to 32*384*384
        self.crop_dataset_path = os.path.join(self.root_path, "crop_slice")
        self.crop_origin_mask = os.path.join(self.crop_dataset_path, "mask")
        self.crop_origin_image = os.path.join(self.crop_dataset_path, "image")
        # for kidney area data may be not normal
        self.crop_cube_path = os.path.join(self.root_path, "kidney_cube")
        self.crop_cube_mask = os.path.join(self.crop_cube_path, "mask")
        self.crop_cube_image = os.path.join(self.crop_cube_path, "image")
        # crop kidney area data to normal shape 64*128*128
        self.crop_64cube_path = os.path.join(self.root_path, "kidney_64cube")
        self.crop_64cube_mask = os.path.join(self.crop_64cube_path, "mask")
        self.crop_64cube_image = os.path.join(self.crop_64cube_path, "image")
        self.coarse = True

        self.crop_xy = (self.area_size[1], self.area_size[2])
        self.case_name_json = os.path.join(self.root_path, "json_file/cases_name.json")
        self.cases_spacing_json = os.path.join(self.root_path, "json_file/cases_spacing.json")
        self.kidney_info_json = os.path.join(self.root_path, "json_file/kidney_info.json")
        self.cube_shape_json = os.path.join(self.root_path, "json_file/cube_shape.json")
        self.cube_name_json = os.path.join(self.root_path, "json_file/cube_name.json")


class CoarseConfig(Config):
    def __init__(self):
        super(CoarseConfig, self).__init__()
        self.layer_thick = 64
        self.area_size = (self.layer_thick, 128, 128)
        self.resize_xy = (128, 128)
        self.crop_xy = (self.area_size[1], self.area_size[2])
        self.coarse = False
        self.batch_size = 16


config = Config()

