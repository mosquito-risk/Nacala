import numpy as np
from data import *
from operator import itemgetter


# estimate metrics according to google sheet order
def estimate_metrics(metrics: list):
    cap50 = [m['cap50'] for m in metrics]
    print(f"Mean CAP50: {np.round(np.mean(cap50, axis=0), 3)}, Std CAP50: {np.round(np.std(cap50, axis=0), 3)}")
    map50 = [m['map50'] for m in metrics]
    print(f"Mean mAP50: {round(np.mean(map50), 3)}, Std mAP50: {round(np.std(map50), 3)}")
    map5095 = [m['map5095'] for m in metrics]
    print(f"Mean mAP5095: {round(np.mean(map5095), 3)}, Std mAP5095: {round(np.std(map5095), 3)}")
    tp_c = [m['tp_c'] for m in metrics]
    print(f"Mean TP_C: {round(np.mean(tp_c), 3)}, Std TP_C: {round(np.std(tp_c), 3)}")
    ap50s = [m['ap50'] for m in metrics]
    print(f"Mean AP50: {round(np.mean(ap50s), 3)}, Std AP50: {round(np.std(ap50s), 3)}")
    ap5095s = [m['ap5095'] for m in metrics]
    print(f"Mean AP5095: {round(np.mean(ap5095s), 3)}, Std AP5095: {round(np.std(ap5095s), 3)}")
    tp = [m['tp'] for m in metrics]
    print(f"Mean TP: {round(np.mean(tp), 3)}, Std TP: {round(np.std(tp), 3)}")

    # Mean AP50 for three classes indice of 0, 1, 4 from ap50s
    ap50s_for_three = np.array([np.array(m['cap50'])[[0, 1, 4]] for m in metrics])
    map50_for_three = np.mean(ap50s_for_three, axis=1)
    print(f"Mean mAP50 (for three): {round(np.mean(map50_for_three), 3)},"
          f" Std mAP50: {round(np.std(map50_for_three), 3)}")

    # Mean AP50 for three classes indice of all classes from ap50s
    ap50s_for_all = np.array([np.array(m['cap50']) for m in metrics])
    map50_for_all = np.mean(ap50s_for_all, axis=1)
    print(f"Mean mAP50 (for all): {round(np.mean(map50_for_all), 3)},"
            f" Std mAP50: {round(np.std(map50_for_all), 3)}")

    ciou = [m['ciou'] for m in metrics]
    # print(ciou)
    print(f"Mean CIoU: {np.round(np.mean(ciou, axis=0), 3)}, Std CIoU: {np.round(np.std(ciou, axis=0), 3)}")
    miou = [m['miou'] for m in metrics]
    print(f"Mean mIoU: {round(np.mean(miou), 3)}, Std mIoU: {round(np.std(miou), 3)}")
    ious = [m['iou'] for m in metrics]
    print(f"Mean IoU: {round(np.mean(ious), 3)}, Std IoU: {round(np.std(ious), 3)}")

    # Mean Iou for three classes indice of 0, 1, 4 from ciou
    ciou_for_three = np.array([np.array(m['ciou'])[[0, 1, 4]] for m in metrics])
    miou_for_three = np.mean(ciou_for_three, axis=1)
    print(f"Mean mIoU (for three): {round(np.mean(miou_for_three), 3)}, Std mIoU: {round(np.std(miou_for_three), 3)}")

# UNet DOW Head1 Test1 (T1)
unet_h1_t1 = [unet_h1_t1_1, unet_h1_t1_2, unet_h1_t1_3, unet_h1_t1_4, unet_h1_t1_5]

# UNet DOW Head2 Test1
unet_h2_t1 = [unet_h2_t1_1, unet_h2_t1_2, unet_h2_t1_3, unet_h2_t1_4, unet_h2_t1_5]

# UNet DOW Head3 Test1
unet_h3_t1 = [unet_h3_t1_1, unet_h3_t1_2, unet_h3_t1_3, unet_h3_t1_4, unet_h3_t1_5]

# UNet DOW Head1 Test2
unet_h1_t2 = [unet_h1_t2_1, unet_h1_t2_2, unet_h1_t2_3, unet_h1_t2_4, unet_h1_t2_5]

# UNet DOW Head2 Test2
unet_h2_t2 = [unet_h2_t2_1, unet_h2_t2_2, unet_h2_t2_3, unet_h2_t2_4, unet_h2_t2_5]

# UNet DOW Head3 Test2
unet_h3_t2 = [unet_h3_t2_1, unet_h3_t2_2, unet_h3_t2_3, unet_h3_t2_4, unet_h3_t2_5]

# DINO DOW Head1 Test1
dino_h1_t1 = [dino_h1_t1_1, dino_h1_t1_2, dino_h1_t1_3, dino_h1_t1_4, dino_h1_t1_5]

# DINO DOW Head2 Test1
dino_h2_t1 = [dino_h2_t1_1, dino_h2_t1_2, dino_h2_t1_3, dino_h2_t1_4, dino_h2_t1_5]

# DINO DOW Head3 Test1
dino_h3_t1 = [dino_h3_t1_1, dino_h3_t1_2, dino_h3_t1_3, dino_h3_t1_4, dino_h3_t1_5]

# DINO DOW Head1 Test2
dino_h1_t2 = [dino_h1_t2_1, dino_h1_t2_2, dino_h1_t2_3, dino_h1_t2_4, dino_h1_t2_5]
# DINO DOW Head2 Test2
dino_h2_t2 = [dino_h2_t2_1, dino_h2_t2_2, dino_h2_t2_3, dino_h2_t2_4, dino_h2_t2_5]
# DINO DOW Head3 Test2
dino_h3_t2 = [dino_h3_t2_1, dino_h3_t2_2, dino_h3_t2_3, dino_h3_t2_4, dino_h3_t2_5]

# UNet Multi Test1
unet_t1 = [unet_t1_1, unet_t1_2, unet_t1_3, unet_t1_4, unet_t1_5]

# DINO Multi Test1
dino_t1 = [dino_t1_1, dino_t1_2, dino_t1_3, dino_t1_4, dino_t1_5]

# UNet Multi Test2
unet_t2 = [unet_t2_1, unet_t2_2, unet_t2_3, unet_t2_4, unet_t2_5]

# DINO Multi Test2
dino_t2 = [dino_t2_1, dino_t2_2, dino_t2_3, dino_t2_4, dino_t2_5]

# UNet Binary Test1 and Test2
unet_t1_b = [unet_t1_b1, unet_t1_b2, unet_t1_b3, unet_t1_b4, unet_t1_b5]
unet_t2_b = [unet_t2_b1, unet_t2_b2, unet_t2_b3, unet_t2_b4, unet_t2_b5]

# UNet DOW Binary Test1 and Test2
unet_dow_t1_b = [unet_dow_t1_b1, unet_dow_t1_b2, unet_dow_t1_b3, unet_dow_t1_b4, unet_dow_t1_b5]
unet_dow_t2_b = [unet_dow_t2_b1, unet_dow_t2_b2, unet_dow_t2_b3, unet_dow_t2_b4, unet_dow_t2_b5]

# DINO Binary Test1 and Test2
dino_t1_b = [dino_t1_b1, dino_t1_b2, dino_t1_b3, dino_t1_b4, dino_t1_b5]
estimate_metrics(dino_t1_b)

