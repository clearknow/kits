"""
    training config
    author: ChuDa xiao
"""
import torch
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
        # resample CT image shape ()
        self.new_size = (128, 256, 256)


class Config(BaseConfig):
    def __init__(self):
        super(Config, self).__init__()
        '''
            set your config 
        '''
        self.batch_size = 16
        self.epochs = 40
        self.spacing = (3.22, 1.62, 1.62)
        self.entropy_weight = [1, 4, 8, 8]
        # dice loss
        # self.loss = 1
        # artery image
        self.artery_image_root = "/public/home/cxiao/Study/data/artery/image"
        self.artery_mask_root = "/public/home/cxiao/Study/data/artery/mask/"
        self.artery_save_path = "/public/home/cxiao/Study/data/artery/predict/"
        self.second_network = False
        # kits
        self.kits_image_root = "/public/home/cxiao/Study/data/kits21/image_2d/"
        self.kits_mask_root = "/public/home/cxiao/Study/data/kits21/mask_2d/"
        self.kits_save_root = "/public/home/cxiao/Study/data/kits21/predict/"
        self.origin_data_path = "/public/datasets/kidney/kits21/kits21/data/"
        self.kits_val_image_path = "/public/home/cxiao/Study/data/kits21/val/image"
        self.kits_val_mask_path = "/public/home/cxiao/Study/data/kits21/val/mask"

        self.image_crop = "/public/home/cxiao/Study/data/kits21/image_crop"
        self.mask_crop = "/public/home/cxiao/Study/data/kits21/mask_crop"
        self.val_image_crop = "/public/home/cxiao/Study/data/kits21/val/image_crop"
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


config = Config()

