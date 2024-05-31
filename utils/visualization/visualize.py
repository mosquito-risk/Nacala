import textwrap
from itertools import product
import re
import matplotlib

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
import seaborn as sns


def visualize_output_of_dataset(dataset, config, class_names, rows=10, offset=0, write_to_pdf=False):
    images_so_far = 0
    if class_names == None:
        class_names = [1]
    cols = len(class_names) + 1
    fig = plt.figure(figsize=(10, 10 * rows // cols))
    N = len(dataset)

    for i in range(offset, N):
        sample = dataset[i]
        dt = image_for_display(sample[config.reference_source][:3])
        images_so_far += 1
        ax = plt.subplot(rows, cols, images_so_far)
        ax.axis('off')
        # import ipdb; ipdb.set_trace()
        img = dt.cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        ax.set_title(
            f'{config.reference_source}: mean: {img.mean():.2f},\n min: {img.min():.2f}, max: {img.max():.2f}')
        plt.imshow(img)

        # Show targets
        dt = sample["target"]
        if len(class_names) == 1:  # Binary segmentation
            images_so_far += 1
            ax = plt.subplot(rows, cols, images_so_far)
            ax.axis('off')
            img = dt.float().cpu().numpy()
            ax.set_title(
                f'target_{class_names[0]}: mean: {img.mean():.2f},\n min: {img.min():.2f}, max: {img.max():.2f}')
            plt.imshow(img)
        else:  # Multiclass segmentation
            for i in range(len(class_names)):
                images_so_far += 1
                ax = plt.subplot(rows, cols, images_so_far)
                ax.axis('off')
                img = (dt == i).float().cpu().numpy()
                ax.set_title(
                    f'target_{class_names[i]}: mean: {img.mean():.2f},\n min: {img.min():.2f}, max: {img.max():.2f}')
                plt.imshow(img)

        if images_so_far == rows * cols:
            plt.show(block=False)

            a = input('Next plot?\n (y/n)')
            if a == "y" or a == "Y":
                plt.close()
                plt.clf()
                images_so_far = 0
            else:
                break

    if write_to_pdf:
        output_fp = f'{config.__name__}.pdf'
        output_fp = output_fp.replace(".pdf", f"{offset}_{offset + images_so_far}.pdf")
        plt.savefig(output_fp)


def image_for_display(img, per_instance: bool = False):
    sq = False
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
        sq = True

    if img.shape[1] > 3:
        img = img[:, :3]
    elif img.shape[1] == 1:
        img = img.expand(-1, 3, -1, -1)
    elif img.shape[1] == 2:
        img = torch.concat((img, img[:, [0]]), dim=1)
    ax = []
    if per_instance:
        ax = [0]
    ax += [2, 3]  # Along the height and width

    nmin = img.amin(dim=ax, keepdim=True)
    nmax = img.amax(dim=ax, keepdim=True)
    i = (img - nmin) / (nmax - nmin)
    if sq:
        i = i.squeeze(0)
    return i


def normalize01(img, channel_axis=[1, 2]):
    imgc = torch.clone(img)
    imgc[imgc == 0] = 999999999
    # Some of the images do not contain data and this is a trick to make sure that, we don't include the zeros values in the img normalization
    nmin = imgc.amin(dim=channel_axis, keepdim=True)
    nmax = img.amax(dim=channel_axis, keepdim=True)
    return (img - nmin) / (nmax - nmin)


def append_pred_to_input(inp_images, gt, mask=None):
    assert type(inp_images) == np.ndarray
    assert type(gt) == np.ndarray
    assert mask is None or type(mask) == np.ndarray

    if inp_images.shape[1] != 1:  # If it's not single channel image, then take a mean along the channels
        inp_images = np.mean(inp_images, 1, keepdims=True)
    inp_images = np.tile(inp_images, (1, 3, 1, 1))  # Make is 3 channel
    inp_images[:, [2]] = gt * 0.5
    # import ipdb; ipdb.set_trace()
    if mask is not None:
        if mask.dtype == np.bool:  # To convert from bool to correct type
            mask = (mask * 0.5)
            mask = mask.astype(inp_images.dtype)
        inp_images[:, 1] = mask
    return inp_images


def image_from_confusion_matrix(confusion_matrix):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = sns.heatmap(confusion_matrix, annot=True, cmap=plt.cm.Blues)
    ax.set(xlabel='Predicted labels', ylabel='True labels')
    fig.canvas.draw()
    # Now we can save it to a numpy array.
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = np.transpose(data, (2, 0, 1))
    data = np.expand_dims(data, axis=0)
    return data


def plot_confusion_matrix(cm, labels):
    '''
    :param cm: A confusion matrix: A square ```numpy array``` of the same size as self.labels
`   :return:  A ``matplotlib.figure.Figure`` object with a numerical and graphical representation of the cm array
    '''
    numClasses = len(labels)
    fig = matplotlib.figure.Figure(figsize=(numClasses*1.6, numClasses*1.6), dpi=100, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(cm, cmap='Oranges')

    classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labels]
    classes = ['\n'.join(textwrap.wrap(l, 20)) for l in classes]

    tick_marks = np.arange(len(classes))

    ax.set_xlabel('Predicted')
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=-90, ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label')
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, va='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in product(range(numClasses), range(numClasses)):
        ax.text(j, i, f"{cm[i, j]:.2f}" if cm[i, j] != 0 else '.', horizontalalignment="center",
                verticalalignment='center', color="black")
    fig.set_tight_layout(True)
    return fig


def add_text_to_image(inp_img, text):
    # Format BCHW, B * Text
    assert type(inp_img) == np.ndarray
    assert len(inp_img.shape) == 4
    from PIL import Image, ImageDraw
    sz = inp_img.shape[2:]
    ti = []
    for i in range(inp_img.shape[0]):
        img = Image.new(mode='RGB', size=sz)
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), text[i], (1, 1, 1))
        img = np.array(img).mean(axis=2)
        ti.append(img)
    ti = np.array(ti)
    ti = np.expand_dims(ti, 1)  # Add the channel dimension
    inp_img = inp_img + ti * 0.8
    return inp_img


def visualize_output_of_dataloader(dataloader, rows=10, cols=5):
    images_so_far = 0
    fig = plt.figure(figsize=(50, 50 * rows // cols))
    for i_batch, sample_batched in enumerate(dataloader):
        sample_keys = [k for k in list(
            sample_batched.keys()) if 'ori_' not in k]
        for j in range(sample_batched[sample_keys[0]].size()[0]):
            for k in sample_keys:
                if 'aerial' in k:
                    images_so_far += 1
                    ax = plt.subplot(rows, cols, images_so_far)
                    ax.axis('off')
                    dt = sample_batched[k]
                    img = dt.cpu().data[j].numpy()
                    ax.set_title(
                        f'{k}: mean: {img.mean():.2f},\n min: {img.min():.2f}, max: {img.max():.2f}')
                    plt.imshow(img.transpose(1, 2, 0))
                else:
                    images_so_far += 1
                    ax = plt.subplot(rows, cols, images_so_far)
                    ax.axis('off')
                    dt = sample_batched[k]
                    img = dt.cpu().data[j].numpy()[0]
                    ax.set_title(
                        f'{k}: mean: {img.mean():.2f},\n min: {img.min():.2f}, max: {img.max():.2f}')
                    plt.imshow(img)

                if images_so_far == rows * cols:
                    return


def visualize_predictions(dataloader, net, starting_batch=0, device='cuda', img_type=torch.float32, rows=10, cols=9):
    images_so_far = 0
    fig = plt.figure(figsize=(50, 50 * rows // cols))
    for i_batch, sample_batched in enumerate(dataloader):
        if i_batch >= starting_batch:
            sample_keys = [k for k in list(
                sample_batched.keys()) if 'ori_' not in k]
            aerial = sample_batched['aerial_20'].to(
                device=device, dtype=img_type)
            with torch.no_grad():
                pred = net(aerial)
            for j in range(sample_batched[sample_keys[0]].size()[0]):
                # Show the input and ground truth
                for k in sample_keys:
                    if 'aerial' in k:
                        images_so_far += 1
                        ax = plt.subplot(rows, cols, images_so_far)
                        ax.axis('off')
                        dt = sample_batched[k]
                        img = dt.cpu().data[j].numpy()
                        ax.set_title(
                            f'{k}: mean: {img.mean():.2f},\n min: {img.min():.2f}, max: {img.max():.2f}')
                        plt.imshow(img.transpose(1, 2, 0))
                    else:
                        images_so_far += 1
                        ax = plt.subplot(rows, cols, images_so_far)
                        ax.axis('off')
                        dt = sample_batched[k]
                        img = dt.cpu().data[j].numpy()[0]
                        ax.set_title(
                            f'{k}: mean: {img.mean():.2f},\n min: {img.min():.2f}, max: {img.max():.2f}')
                        plt.imshow(img)
                    # Show the predictions
                for k in range(pred.shape[1]):
                    images_so_far += 1
                    ax = plt.subplot(rows, cols, images_so_far)
                    ax.axis('off')
                    if k == 0:
                        img = F.sigmoid(pred).cpu().data[j, k].numpy()
                    else:
                        img = pred.cpu().data[j, k].numpy()
                    ax.set_title(
                        f'{k}: mean: {img.mean():.2f},\n min: {img.min():.2f}, max: {img.max():.2f}')
                    plt.imshow(img)

            if images_so_far == rows * cols:
                return
