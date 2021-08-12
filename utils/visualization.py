"""
    display the image
    3 views: show_views()
"""
import matplotlib.pyplot as plt
import numpy as np


def show_views(image, mask, prob=None, cmap=None, image_title=["image", "mask", "predict"]):
    """
    show 2 or 3 image image
    :param image: tensor or numpy h *w
    :param mask:   tensor or numpy h *w
    :param prob:  tensor or numpy h *w
    :param cmap:  color :gray
    :param image_title: default ["image", "mask", "predict"]
    :return:
    """
    counter = 2
    image_list = [image, mask]
    # title = ["image", "mask", "predict"]
    if prob is not None:
        counter = 3
        image_list.append(prob)

    for i in range(0, counter):
        ax1 = plt.subplot(1, counter, i+1)
        ax1.set_title(image_title[i])
        ax1.axis("off")
        ax1.imshow(image_list[i], cmap=cmap)
    plt.axis("off")
    plt.show()


def show_single_view(image, cmap=None):
    plt.imshow(image, cmap=cmap)
    plt.axis("off")
    plt.show()


def show_image_hist(image):
    plt.hist(image)
    plt.title("histogram")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    import nibabel as nib
    root_image = "/public/datasets/kidney/kits21/kits21/data/case_00000/imaging.nii.gz"
    root_mask = "/public/datasets/kidney/kits21/kits21/data/case_00000/aggregated_AND_seg.nii.gz"
    images = nib.load(root_image).get_data()
    images[images < -79] = 0
    images[images > 304] = 0

    print(images.max(), images.min())
    show_image_hist(images[100])
    show_single_view(images[100])
