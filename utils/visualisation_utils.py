import matplotlib.pyplot as plt
import rasterio
import random


def pr_curve(precision, recall):
    # Start plotting
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.', label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_poly_dataframes(dataframes: list):
    if len(dataframes) == 1:
        dataframes[0].plot()
    elif len(dataframes) == 2:
        ax1 = dataframes[0].boundary.plot(color='r')
        dataframes[1].plot(ax=ax1)
    plt.show()


def plot_image_and_polygons(image_path, dataframe):
    fig, ax = plt.subplots(figsize=(15, 15))
    with rasterio.open(image_path) as src:
        rgb = src.read((1, 2, 3))
    ax.imshow(rgb.transpose(1, 2, 0))
    random_colors = ['#' + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for _ in range(len(dataframe))]
    dataframe.boundary.plot(ax=ax, color=None, edgecolor=random_colors, linewidth=2)
    plt.show()
