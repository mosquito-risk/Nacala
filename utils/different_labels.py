import numpy as np
from scipy.ndimage import binary_erosion
from scipy import ndimage as ndi
from skimage.segmentation import watershed


def watershed_image(input_array):
    """
    Create watershed image from energy levels
    :param input_array:
    :return: watershed image
    """
    input_array = input_array.astype(np.uint16)
    binary_img = input_array > 0
    return watershed(-input_array, connectivity=1, mask=binary_img)


def single_output_from_multihead(head1, head2):
    """
    Create watershed image on input_array1 using input_array2 as markers
    :param head1: first head's output of the model
    :param head2: second head's output of the model
    :return: watershed image
    """
    input_dist = head2 == 0
    distance = ndi.distance_transform_edt(input_dist)
    distance = np.where(head1 == 1, distance, 0).astype(np.uint16)
    return watershed(distance, connectivity=2, mask=head1)

def single_output_from_multihead_cls(head1, head2):
    """
    Create watershed image on input_array1 using input_array2 as markers
    :param head1: first head's output of the model
    :param head2: second head's output of the model
    :return: watershed image
    """
    input_dist = head2 == 0
    distance = ndi.distance_transform_edt(input_dist)
    distance = np.where(head1 == 1, distance, 0).astype(np.uint16)
    return watershed(distance, connectivity=2, mask=head1)


def get_border_and_interior_of_binary_mask(mask, border_width=4):
    # Erode the mask
    eroded_mask = binary_erosion(mask, iterations=border_width)
    # The border is the difference between the dilated and eroded masks
    border = mask - eroded_mask
    return border, eroded_mask


def int_ext_labels(final_label):
    int_mask = np.zeros_like(final_label)
    ext_mask = np.zeros_like(final_label)
    final_label_binary = np.where(final_label > 0, 1, 0)
    labeled_image, objects_count = ndi.label(final_label_binary)
    if objects_count != 0:
        for i in range(1, objects_count + 1):
            object_mask = np.where(labeled_image == i, 1, 0)
            border, interior = get_border_and_interior_of_binary_mask(object_mask)
            int_mask = np.maximum(int_mask, interior)
            ext_mask = np.maximum(ext_mask, border)
    return int_mask, ext_mask
