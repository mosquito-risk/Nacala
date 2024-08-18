import json
import os
import logging
import cv2
import copy
from pycocotools import mask
import numpy as np
from itertools import groupby

# category list for different datasets
data_fusion_contest = [{"supercategory": "flat_roof", "id": 1, "name": "flat_roof"},
                       {"supercategory": "gable_roof", "id": 2, "name": "gable_roof"},
                       {"supercategory": "gambrel_roof", "id": 3, "name": "gambrel_roof"},
                       {"supercategory": "row_roof", "id": 4, "name": "row_roof"},
                       {"supercategory": "multiple_eave_roof", "id": 5, "name": "multiple_eave_roof"},
                       {"supercategory": "hipped_roof_v1", "id": 6, "name": "hipped_roof_v1"},
                       {"supercategory": "hipped_roof_v2", "id": 7, "name": "hipped_roof_v2"},
                       {"supercategory": "mansard_roof", "id": 8, "name": "mansard_roof"},
                       {"supercategory": "pyramid_roof", "id": 9, "name": "pyramid_roof"},
                       {"supercategory": "arched_roof", "id": 10, "name": "arched_roof"},
                       {"supercategory": "dome", "id": 11, "name": "dome"},
                       {"supercategory": "other", "id": 12, "name": "other"}]

nacala_drone_imagery = [{"supercategory": "thatch", "id": 2, "name": "thatch"},
                        {"supercategory": "metal_sheet", "id": 1, "name": "metal_sheet"},
                        {"supercategory": "concrete", "id": 4, "name": "concrete"},
                        {"supercategory": "asbestos", "id": 3, "name": "asbestos"},
                        {"supercategory": "no_roof", "id": 5, "name": "no_roof"}]

nacala_drone_imagery_binary = [{"supercategory": "building", "id": 1, "name": "building"}]

tree_points = [{"supercategory": "tree", "id": 1, "name": "tree"}]


def segment_area(segment):
    n = len(segment) // 2
    if n < 3:
        return 0
    area = 0
    for i in range(n - 1):
        x1, y1 = segment[2 * i], segment[2 * i + 1]
        x2, y2 = segment[2 * i + 2], segment[2 * i + 3]
        # Shoelace formula
        area += (x1 * y2) - (x2 * y1)
    x1, y1 = segment[-2], segment[-1]
    x2, y2 = segment[0], segment[1]
    area += (x1 * y2) - (x2 * y1)

    area = abs(area) / 2
    return area


def save_coco_json(filename, dictionary):
    with open(filename, 'w', encoding='utf-8') as f:
        # Serialize the data and write it to the file
        json.dump(dictionary, f, indent=4)


# converting rle to list of list
def rle_to_coco(annotation: dict) -> list[dict]:
    """Transform the rle coco annotation (a single one) into coco style.
    In this case, one mask can contain several polygons, later leading to several `Annotation` objects.
    In case of not having a valid polygon (the mask is a single pixel) it will be an empty list.
    source code: https://stackoverflow.com/questions/75326066/coco-annotations-convert-rle-to-polygon-segmentation
    Parameters
    ----------
    annotation : dict
        rle coco style annotation
    Returns
    -------
    list[dict]
        list of coco style annotations (in dict format)
    """

    maskedArr = mask.decode(annotation["segmentation"])
    contours, _ = cv2.findContours(maskedArr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []

    for contour in contours:
        if contour.size >= 6:
            segmentation.append(contour)

    if len(segmentation) == 0:
        logging.debug(
            f"Annotation with id {annotation['id']} is not valid, it has no segmentations."
        )
        annotations = []

    else:
        annotations = list()
        for i, seg in enumerate(segmentation):
            single_annotation = copy.deepcopy(annotation)
            single_annotation["segmentation_coords"] = (
                seg.astype(float).flatten().tolist()
            )
            single_annotation["bbox"] = list(cv2.boundingRect(seg))
            single_annotation["area"] = cv2.contourArea(seg)
            single_annotation["instance_id"] = annotation["id"]
            single_annotation["annotation_id"] = f"{annotation['id']}_{i}"

            annotations.append(single_annotation)

    return annotations


def varify_ref_json(json_filename):
    """
    Pycocotools considers polygon id in ground truth start from 1. This function verify if polygon id start from 1 or 0.
    If id start from 0, all values of polygon ids added by 1
    :param json_filename: JSON file with annotations
    :return: JSON file with polygon ids start from 1
    """
    # load json file
    f = open(json_filename)
    data = json.load(f)

    # veify minimum value of polygon id
    anns = data["annotations"]
    ids = [int(_ann['id']) for _ann in anns]
    min_id = min(ids)
    if min_id == 0:
        org_after_bugfix = []
        for ann in anns:
            ann_ = ann
            ann_["id"] = ann_["id"] + 1
            org_after_bugfix.append(ann_)

        bugfix_json_data = {
            "images": data["images"],
            "annotations": org_after_bugfix,
            "categories": data["categories"]
        }

        json_filename = json_filename[:-5] + '_pid' + json_filename[-5:]
        with open(json_filename, 'w', encoding='utf-8') as f:
            # Serialize the data and write it to the file
            json.dump(bugfix_json_data, f)
    return json_filename


def custom_json(images, ref_json, outfile_name, covert_rle=False, subtract=False):
    """
    Custom JSON file or Subset of JSON for target images:
    * Reference JSON should have file_name attribute and image name from images should match
    :param subtract: Subtract 1 from labels (This is used when there is no background class)
    :param images: List of images
    :param ref_json: JSON file that consist of images and their annotations
    :param outfile_name: Output filename
    :param covert_rle: Convert Run Lenght Encoding (RLE) to list of coordinates
    :return: Subset of JSON with annotations for required images
    """

    # verify reference json file
    ref_json = varify_ref_json(ref_json)

    final_image_dicts = []
    final_annotation_dicts = []
    ref_json = open(ref_json)
    ref_json_data = json.load(ref_json)
    all_image_dicts = ref_json_data['images']
    print(f'There are {len(images)} in the validation set')
    print(f"There are {len(all_image_dicts)} images in the ref json file")
    height, width = 512, 512
    multi_polygons = 0
    i_ = 1
    for image in images:
        image_name = os.path.basename(image)
        image_ids = [dict_['id'] for dict_ in all_image_dicts if dict_['file_name'] == image_name]
        assert len(image_ids) == 1, f"Leght of images id should be 1, but there are {len(image_ids)}"
        # image_dictionary
        image_dict = {"file_name": image_name, "height": height, "width": width, "id": int(image_ids[0])}
        final_image_dicts.append(image_dict)

        # converting required images segments in to rle and creating json
        for ann in ref_json_data['annotations']:  # this has to optimise and reduce iterations
            if ann['image_id'] == int(image_ids[0]):
                if len(ann["segmentation"]) == 1:  # segment as list of coordinates
                    if covert_rle:
                        rle = mask.merge(mask.frPyObjects(ann['segmentation'], 512, 512))
                        rle = {'size': [512, 512], 'counts': rle['counts'].decode("utf-8")}
                    else:
                        rle = ann['segmentation']
                elif type(ann["segmentation"]) == dict:  # segment as rle dictionary
                    if covert_rle:
                        rle = ann["segmentation"]
                    else:
                        new_ann = rle_to_coco(ann)
                        # rle = mask.decode(mask.frPyObjects([new_ann[0]["segmentation_coords"]], 512, 512))
                        rle = [new_ann[0]["segmentation_coords"]]

                else:
                    if covert_rle:
                        rle = mask.merge(mask.frPyObjects(ann["segmentation"], 512, 512))
                        rle = {'size': [512, 512], 'counts': rle['counts'].decode("utf-8")}
                    else:
                        rle = ann["segmentation"]

                if subtract:
                    cat_id = ann['category_id'] - 1
                else:
                    cat_id = ann['category_id']
                # annotation dictionary
                annotation_dict = {
                    "segmentation": rle,
                    'iscrowd': 0,
                    "image_id": int(image_ids[0]),
                    "category_id": cat_id,
                    "bbox": ann['bbox'],
                    "area": ann['area'],
                    "id": ann['id']
                }
                final_annotation_dicts.append(annotation_dict)

        i_ += 1
        print(i_)
    print(f'Number of multi polygons in the data are: {multi_polygons}')
    final_category_dict = ref_json_data['categories']

    final_json_dict = {
        "images": final_image_dicts,
        "annotations": final_annotation_dicts,
        "categories": final_category_dict
    }
    # Writing to sample.json
    save_coco_json(outfile_name, final_json_dict)


def get_coords(poly):
    """
    Function to save each polygon coords in a list of coords [[(x1,y1),(x2,y2),....(x1,y1)]]
    for multi-polygon [[(x1,y1),(x2,y2),....(x1,y1)], [(x1,y1),(x2,y2),....(x1,y1)]]
    :param poly: shapely polygon
    :return: list of coo
    """
    if poly.geom_type == 'MultiPolygon':
        xy = []
        for geom in poly.geoms:
            xy_ = list(geom.exterior.coords)
            xy.append(xy_)
        return xy
    elif poly.geom_type == 'Polygon':
        xy_ = list(poly.exterior.coords)
        return [xy_]
    else:
        print("Not polygon")


def coord_to_rowcol(coord_list, transform):
    """
    Covert coordinates to rows and columns
    :param coord_list: list of coordinates in [[(x1,y1),(x2,y2),....(x1,y1)]] format
    :param transform: gdal transform matrix
    :return:
    """
    ulx, xres, xskew, uly, yskew, yres = transform
    assert xskew == 0 and yskew == 0, 'The roation of image/patch is not zero: coord_to_rowcol function has to modify'
    new_coords = []
    for i in coord_list:
        _x_ = abs(i[0] - ulx) / abs(xres)
        _y_ = abs(i[1] - uly) / abs(yres)
        new_coords += [_x_, _y_]
    return new_coords


def single_coord_to_rowcol(row, transform):
    """
    Covert single coordinate0 to row and column
    :param single_coord: single coordinate in [(x1,y1)] format
    :param transform: gdal transform matrix
    :return:
    """
    ulx, xres, xskew, uly, yskew, yres = transform
    assert xskew == 0 and yskew == 0, 'The roation of image/patch is not zero: coord_to_rowcol function has to modify'
    single_coord = [row.geometry.x, row.geometry.y]
    return [abs(single_coord[0] - ulx) / abs(xres), abs(single_coord[1] - uly) / abs(yres)]


def bbox_from_rowcol(rowcols: list) -> list:
    """
    Function for getting bbox from list of coordinates in rows and cols
    coco bbox format: [xmin, ymin, width, height]
    :param rowcols: list of coordinates
    :return:
    """
    row, col = [r_ for idx, r_ in enumerate(rowcols) if idx % 2 == 0], \
        [r_ for idx, r_ in enumerate(rowcols) if idx % 2 != 0]
    xmin, xmax, ymin, ymax = min(row), max(row), min(col), max(col)
    return [xmin, ymin, xmax - xmin, ymax - ymin]


def coco_annotation_dict(dataframe, image_id, transform, label_attribute=None, start_ann_id=None,
                         extra_attr=None, ann_type='gt'):
    """
    Coco annotation dictionary for single paatch/image
    :param ann_type:
    :param extra_attr:
    :param dataframe: dataframe for single image/patch
    :param label_attribute:
    :param image_id:
    :param transform:
    :param start_ann_id:
    :return:
    """
    annotation_dicts = []
    for idx, row in dataframe.iterrows():
        coords = get_coords(row.geometry)
        if label_attribute is not None:
            cat_id = row[label_attribute]
        else:
            cat_id = 0
        for poly_coord in coords:
            poly_coord_ = coord_to_rowcol(poly_coord, transform)
            bbox_ = bbox_from_rowcol(poly_coord_)

            annotation_dict = {
                "segmentation": [poly_coord_],
                "image_id": image_id,
                "category_id": cat_id,  # fixme this has to be cat_id but changed for binary segmentation
                "iscrowd": 0,
                "bbox": bbox_  # These can be derived using pycocotools from segmentation
            }
            if extra_attr is not None:
                annotation_dict[extra_attr] = row[extra_attr]
            if ann_type == 'gt':
                ann_area = segment_area(poly_coord_)
                annotation_dict["area"] = ann_area
                annotation_dict["id"] = start_ann_id
                start_ann_id += 1
            elif ann_type == 'dt':
                annotation_dict["score"] = row['score']
            # print(annotation_dict)
            annotation_dicts.append(annotation_dict)
    return annotation_dicts, start_ann_id


def coco_annotation_dict_for_point_labels(dataframe, image_id, transform, label_attribute=None, start_ann_id=None,
                                          ann_type='gt'):
    """
    Coco annotation dictionary for single paatch/image
    :param ann_type:
    :param extra_attr:
    :param dataframe: dataframe for single image/patch
    :param label_attribute:
    :param image_id:
    :param transform:
    :param start_ann_id:
    :return:
    """
    annotation_dicts = []
    for idx, row in dataframe.iterrows():
        if label_attribute is not None:
            cat_id = row[label_attribute]
        else:
            cat_id = 0
        point_coord = single_coord_to_rowcol(row, transform)

        annotation_dict = {
            "segmentation": [point_coord],
            "image_id": image_id,
            "category_id": cat_id,  # fixme this has to be cat_id but changed for binary segmentation
            "iscrowd": 0
        }

        if ann_type == 'gt':
            annotation_dict["id"] = start_ann_id
            start_ann_id += 1
        elif ann_type == 'dt':
            annotation_dict["score"] = row['score']

        annotation_dicts.append(annotation_dict)
    return annotation_dicts, start_ann_id


def custom_coco_from_polygons(final_image_dicts, final_annotation_dicts, dataset_name='nacala_binary'):
    if isinstance(dataset_name, str):
        if dataset_name == 'nacala':
            final_category_dicts = nacala_drone_imagery
        elif dataset_name == 'nacala_binary':
            final_category_dicts = nacala_drone_imagery_binary
        elif dataset_name == 'data_fusion':
            final_category_dicts = data_fusion_contest
        elif dataset_name == 'spacenet_sample':
            final_category_dicts = nacala_drone_imagery_binary
        elif dataset_name == 'tree_points':
            final_category_dicts = tree_points
        else:
            raise ValueError(f"Dataset name {dataset_name} is not valid")
    elif isinstance(dataset_name, list):
        final_category_dicts = dataset_name
    else:
        raise ValueError(f"Dataset name {dataset_name} is not valid")

    final_json_dict = {
        "images": final_image_dicts,
        "annotations": final_annotation_dicts,
        "categories": final_category_dicts
    }
    return final_json_dict


def read_json_data(json_file):
    json_data = open(json_file)
    return json.load(json_data)


def write_json_data(filename, data):
    with open(filename, 'w', encoding='utf-8') as f:
        # Serialize the data and write it to the file
        json.dump(data, f)


# create a ground truth and detection files for single images
def mini_json_files(gt, dt, image_id):
    gt_data = read_json_data(gt)
    dt_data = read_json_data(dt)

    # create gt file for single images
    gt_filename = f"../data/coco/{image_id}_gt.json"
    # take annotations from the images and assign ids from 1 to n
    anns = [x for x in gt_data['annotations'] if x["image_id"] == image_id]
    id = 1
    for ann in anns:
        ann["id"] = id
        id += 1
    final_json_gt = {
        "images": [x for x in gt_data['images'] if x["id"] == image_id],
        "annotations": anns,
        "categories": gt_data["categories"]
    }
    write_json_data(gt_filename, final_json_gt)

    # create dt file for single images
    dt_filename = f"../data/coco/{image_id}_dt.json"
    final_json_dt = [x for x in dt_data if x["image_id"] == image_id]
    write_json_data(dt_filename, final_json_dt)


def print_evalImages(eval_images_list):
    print("Number of evalImags: ", len(eval_images_list))
    # Number of dictionaries in the evalImags equal to (number of total categories x area range (eg: all, small, medium,
    # large), The code has to modify to get all
    for img in eval_images_list:
        if img is not None:  # img is None if there is no category in the ground truth or detections
            print(f"Image id: {img['image_id']}, Category Id: {img['category_id']}, Area Range: {img['aRng']},"
                  f"Detection Ids:  {img['dtIds']}, Ground Thuth IDs: {img['gtIds']}")


def coco_ap50_score(cocoEval_object, image_id):
    cocoEval_object.params.maxDets = [10000000]
    cocoEval_object.params.imgIds = [image_id]
    cocoEval_object.evaluate()
    cocoEval_object.accumulate()
    cocoEval_object.summarize_2()
    ap_ = cocoEval_object.stats[0]
    return ap_


def ap50_per_class(cocoEval_object, class_list: list, image_id: int = None):
    cocoEval_object.params.maxDets = [10000000]
    cocoEval_object.params.iouThrs = [0.5]
    if image_id:
        cocoEval_object.params.imgIds = [image_id]
    else:
        # If no image ID is provided, evaluate on all images
        cocoEval_object.params.imgIds = cocoEval_object.cocoGt.getImgIds()
    ap50_list = []
    for cls in class_list:
        cocoEval_object.params.catIds = [cls]
        cocoEval_object.evaluate()
        cocoEval_object.accumulate()
        cocoEval_object.summarize_2()
        ap50_list.append(cocoEval_object.stats[0])
    return ap50_list


def ap50_per_thresh(cocoEval_object, image_id: int):
    cocoEval_object.params.maxDets = [10000000]
    cocoEval_object.params.imgIds = [image_id]
    iou_thresholds = [round(i, 2) for i in list(np.arange(0.5, 1.0, 0.05))]
    ap50_list = []
    for iou in iou_thresholds:
        cocoEval_object.params.iouThrs = [iou]
        cocoEval_object.evaluate()
        cocoEval_object.accumulate()
        cocoEval_object.summarize_2()
        ap50_list.append(cocoEval_object.stats[0])
    return ap50_list


def print_parameters(cocoDt, cocoEval):
    # Print how many detections and ground truth objects are there in the
    print("------------------------------------------------------------------------")
    print("Number of objects in the detections: ", len(cocoDt.anns))
    print("List of category ids in the data: ", cocoEval.params.catIds)
    print("Deafualt IOU Threshold: ", cocoEval.params.iouThrs)
    print("Area range of polygons: ", cocoEval.params.areaRng)
    print("Maximum detections in the images: ", cocoEval.params.maxDets)


def out_annotation_dict(objects_image, image_id, roof_class, score_image=None, object_score=None):
    # get confidence score from confidence score
    if object_score is None:
        assert score_image is not None
        object_score = float(np.sum(score_image[objects_image]) / np.sum(objects_image))
    # object_score = float(np.max(score_image[objects_image]))
    # import ipdb; ipdb.set_trace()
    rle = binary_mask_to_rle(objects_image)
    compressed_rle = mask.frPyObjects(rle, rle.get('size')[0], rle.get('size')[1])
    # import ipdb
    # ipdb.set_trace()
    # print(mask.toBbox(rle))

    compressed_rle['counts'] = compressed_rle['counts'].decode("utf-8")
    annotation_dict = {"image_id": int(image_id),
                       "score": object_score,
                       "category_id": int(roof_class),
                       "segmentation": compressed_rle,
                       "bbox": list(mask.toBbox(compressed_rle))}
    return annotation_dict


def binary_mask_to_rle(binary_mask):
    binary_mask = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle
