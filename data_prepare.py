import json
from pipelines import patchgen_fcn_object

# data parameterrs

params = {
    'image_dir': './datasets/nacala/train/',
    'label_dir': './datasets/nacala/train/',
    'out_folder': './datasets/nacala/output/',
    'patch_size': 512,
    'overlap': 0,
    'label_attribute': 'mater_id',
    'write_images': True,
    'coco_labels': True,
    'yolo_labels': True,
    'yolo_binary': True,
    'patch_labels': True,
    'geojson_labels': False,
    'int_mask': False,
    'inter_per': 50,
    'weight_mask': True,
    'energy_mask': False,
    'level_dist': 3,
    'int_mask_euc': True,
    'exterior_dist': 10,
    'coco_category_dict': 'nacala',
    'w0': 10,
    'sigma': 5,
    'label_format': 'geojson',
    'data_per_thresh': 0.4,
    'subset_info': None
}

# Create object of the class
print("Parameters: \n", json.dumps(params, indent=4))

fcn_datagen = patchgen_fcn_object.patch_gen(**params)
fcn_datagen.run_datagen_pipeline()
