import json
from pipelines import patchgen_fcn_object

# data parameterrs

params = {
    'image_dir': '/scratch/project_465002161/projects/Nacala/datasets/raw_data/valid',
    'label_dir': '/scratch/project_465002161/projects/Nacala/datasets/raw_data/valid',
    'out_folder': '/scratch/project_465002161/projects/Nacala/datasets/ablation/valid',
    'patch_size': 512,
    'overlap': 0,
    'label_attribute': 'mater_id',
    'write_images': False,
    'coco_labels': False,
    'yolo_labels': False,
    'yolo_binary': False,
    'patch_labels': False,
    'geojson_labels': False,
    'int_mask': False,
    'inter_per': 50,
    'weight_mask': False,
    'energy_mask': False,
    'level_dist': 3,
    'int_mask_euc': True,
    'exterior_dist': 5,
    'coco_category_dict': 'nacala',
    'w0': 10,
    'sigma': 5,
    'label_format': 'geojson',
    'data_per_thresh': 0.4,
    # 'subset_info': '../data/nacala_dataset/raw_data/scaling_files/scaling_law20p.shp'
}

# save params as json
# with open(f'/scratch/project_465001005/projects/nacala/mar5/patchgen_params_yolob_train.json', 'w') as file:
#     json.dump(params, file, indent=4)

# Create object of the class
print("Parameters: \n", json.dumps(params, indent=4))

fcn_datagen = patchgen_fcn_object.patch_gen(**params)
fcn_datagen.run_datagen_pipeline()
