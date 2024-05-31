import numpy as np
from utils.visualization.visualize import plot_confusion_matrix


def write_cm(writer, num_classes, cmat, set_type='train', id=0):
    # Done this way, since not sure how normalize = True effect mious etc
    cmatn = cmat / cmat.sum(axis=1, keepdim=True)
    cmatn = np.nan_to_num(cmatn.cpu().numpy())
    if num_classes == 2:
        class_names = ['non-buildings', 'buildings']
    elif num_classes == 6:
        class_names = ['non-buildings', 'metal_sheet', 'thatch', 'asbestos', 'concrete', 'no_roof']
    elif num_classes == 13:
        class_names = ['non-buildings', 'flat', 'gable', 'gambrel', 'row', 'multiple\neave',
                       'hipped\nv1', 'hipped\nv2', 'mansard', 'pyramid', 'arched', 'dome', 'other']
    else:
        class_names = [f'class_{i}' for i in range(num_classes)]
    writer.add_figure(f"{set_type}/confusion_matrix_{id}", plot_confusion_matrix(cmatn, class_names))
