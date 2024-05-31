import rasterio
import numpy as np
from rasterio.features import geometry_mask
from rasterio.features import rasterize
from rasterio.windows import Window
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import Polygon


def new_row_column_offsets(col_off, row_off, height, width, patch_type='constant'):
    """
    Creating new patch corrdinates by centering the object
    """
    if patch_type == 'constant':
        patch_size = 448
    else:
        patch_size = max(round(height), round(width))

    if patch_size > width:
        col_off = col_off - abs(patch_size - width) / 2
    else:
        col_off = col_off + abs(patch_size - width) / 2

    if patch_size > height:
        row_off = row_off - abs(patch_size - height) / 2
    else:
        row_off = row_off + abs(patch_size - height) / 2

    return col_off, row_off, patch_size


def bbox_to_window(bbox):
    """
    Convert a bounding box to a Rasterio window.
    :param bbox: A tuple of slices representing the bounding box.
    :return: A Rasterio window.
    """
    row_start, row_stop = bbox[0].start, bbox[0].stop
    col_start, col_stop = bbox[1].start, bbox[1].stop
    width = col_stop - col_start
    height = row_stop - row_start
    col_off, row_off, patch_size = new_row_column_offsets(col_start, row_start, height, width)
    window = Window(col_off=col_off, row_off=row_off, width=patch_size, height=patch_size)
    return window


# function for creating mask using dataframa and tif file
def create_mask(dataframe, image):
    with rasterio.open(image) as src:
        transform = src.transform
        geom_list = [brow[1].geometry for brow in dataframe.iterrows()]
        mask = geometry_mask(geom_list, transform=transform, invert=False,
                             out_shape=(src.width, src.height))
        return mask


# A function to burn a GeoDataFrame's geometry into a raster.
def burn_geometry_to_raster(gdf, image, label_attribute):
    # Ensure the label column is integer type
    gdf[label_attribute] = gdf[label_attribute].astype(int)

    # get meta data from image
    with rasterio.open(image) as src:
        image_meta = src.meta

    # Modify the metadata to have one single band and to be of type integer.
    meta = image_meta.copy()
    meta.update(compress='lzw', dtype=rasterio.int32, count=1)

    # Create a blank raster with same shape as `image_meta`.
    burned_raster = np.zeros((image_meta['height'], image_meta['width']), dtype=np.int32)

    # Burn the values
    burned_raster = rasterize(
        shapes=zip(gdf.geometry, gdf[label_attribute].values),
        out_shape=(image_meta['height'], image_meta['width']),
        transform=image_meta['transform'],
        fill=0,  # Fill the areas outside of the shapes with 0.
        default_value=1,  # Burn 1 for all true geometry in raster.
        all_touched=True,  # Consider all pixels touched by geometries.
        dtype=rasterio.int32
    )
    return burned_raster


def window_to_polygon(transform, window):
    """
    Convert a rasterio window to a polygon in the dataset's coordinate system.

    :param transform: Affine transform of the rasterio dataset.
    :param window: rasterio window object.
    :return: shapely Polygon in the dataset's coordinate system.
    """
    # Calculate the bounds of the window
    row_start, col_start = window.row_off, window.col_off
    row_stop = row_start + window.height
    col_stop = col_start + window.width

    # Use the affine transform to convert window corners to geographic coordinates
    ul = transform * (col_start, row_start)  # Upper left
    ur = transform * (col_stop, row_start)   # Upper right
    lr = transform * (col_stop, row_stop)    # Lower right
    ll = transform * (col_start, row_stop)   # Lower left

    # Create a polygon from these coordinates
    polygon = Polygon([ul, ur, lr, ll, ul])

    return polygon


def get_transform(image_path):
    with rasterio.open(image_path) as dataset:
        return dataset.transform


def rasterize_gdf_within_window(window, transform, gdf, label_attribute):
    """
    Rasterize polygons from a GeoDataFrame within a specified window into a NumPy array.

    :param window: rasterio window object.
    :param transform: Affine transform for the window.
    :param gdf: GeoDataFrame containing polygon geometries with labels.
    :return: NumPy array with the rasterized labels.
    """
    # Get the transformed shapes and their corresponding labels
    shapes_and_labels = ((geom, label) for geom, label in zip(gdf.geometry, gdf[label_attribute]))

    # Rasterize the polygons
    rasterized_array = rasterize(
        shapes_and_labels,
        out_shape=(window.height, window.width),
        transform=transform,
        fill=0,
        all_touched=True
    )

    return rasterized_array
