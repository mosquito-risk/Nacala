from shapely.ops import cascaded_union, snap
from shapely.geometry import MultiPolygon, Polygon, box
import geopandas as gpd
import warnings
from geopandas.tools import overlay
import os

warnings.filterwarnings("ignore")


def get_xy(poly, coco=False):
    if poly.geom_type == 'MultiPolygon':
        x, y = [], []
        for geom in poly.geoms:
            x_, y_ = geom.exterior.xy
            x, y = x + list(x_), y + list(y_)
        return x, y
    elif poly.geom_type == 'Polygon':
        x, y = poly.exterior.xy
        return x, y
    else:
        print("Not polygon")


def patch_bbox(transform, xoff, yoff, patch_size):
    ulx, xres, xskew, uly, yskew, yres = transform
    minx, miny, maxx, maxy = ulx + xoff * xres, uly + yoff * yres, ulx + xoff * xres + patch_size * xres, \
                             uly + yoff * yres + patch_size * yres
    return box(minx, miny, maxx, maxy)


def patch_transform(transform, xoff, yoff):
    ulx, xres, xskew, uly, yskew, yres = transform
    patch_ulx, patch_uly = ulx + xoff * xres, uly + yoff * yres
    patch_trans = (patch_ulx, xres, xskew, patch_uly, yskew, yres)
    return patch_trans


def clip_polygon(small_poly, big_poly):
    """
    Clip data using polygon
    :param small_poly: shapely polygon
    :param big_poly: geopandas dataframe
    :return: big_poly within small_poly
    """
    return gpd.clip(big_poly, small_poly)


def intersection(polygon1, polygon2, area=False):
    """
    :param polygon1: shapely polygon
    :param polygon2: shapely polygon
    :param area: area
    :return: Calculates area of intersection polygon if area is true else returns intersection polygon
    """
    intersection_poly = polygon1.intersection(polygon2)
    if area:
        return intersection_poly.area
    else:
        return intersection_poly


def union(polygon1, polygon2, area=False):
    """
    :param polygon1: shapely polygon
    :param polygon2: shapely polygon
    :param area: area
    :return: Calculates area of union polygon if area is true else returns intersection polygon
    """
    union_poly = cascaded_union([polygon1, polygon2])
    if area:
        return union_poly.area
    else:
        return union_poly


def merge_intersecting_polygons(dataframe, io_small_area_th=0.1):
    new_polygon_list = []
    processed_polygon_list = []
    invalid_geometries = dataframe[~dataframe.is_valid]
    # If there are invalid geometries, try fixing them
    if not invalid_geometries.empty:
        dataframe['geometry'] = dataframe.apply(
            lambda row: row['geometry'].buffer(0) if not row['geometry'].is_valid else row['geometry'], axis=1)
    dataframe['area'] = dataframe['geometry'].area
    dataframe = dataframe.sort_values(by='area', ascending=False).reset_index()
    # import ipdb; ipdb.set_trace()
    for index, row in dataframe.iterrows():
        if row.id not in processed_polygon_list:
            gdf1 = dataframe.copy()
            # all polygons except the current polygon
            gdf1 = gdf1[gdf1.id != row.id].reset_index(drop=True)
            intersecting_polygon_loc = gdf1.intersects(row.geometry)
            intersecting_polygons = gdf1[intersecting_polygon_loc]
            temp_polygon = row.geometry
            if intersecting_polygons.shape[0] > 0:
                # print(f'Intersecting polygons {intersecting_polygons.shape[0]}')
                for index2, polygon2 in intersecting_polygons.iterrows():
                    area2 = polygon2.geometry.area
                    area1 = temp_polygon.area
                    intersection = row.geometry.intersection(polygon2.geometry)
                    io_small_area = intersection.area / min(area2, area1)
                    if io_small_area >= io_small_area_th:
                        temp_polygon = cascaded_union([temp_polygon, polygon2.geometry])
                        dataframe = dataframe[dataframe.id != polygon2.id]
                        processed_polygon_list.append(polygon2.id)
                new_polygon_list.append([index, temp_polygon])
                processed_polygon_list.append(row.id)
        # print(processed_polygon_list)
    print("Toatal number of polygons:", len(new_polygon_list))
    df = gpd.GeoDataFrame(new_polygon_list, columns=['_id', 'geometry'])
    return df


def merge_small_polygons(dataframe, threshold=0.1):
    dataframe['area'] = dataframe['geometry'].area
    gdf = dataframe.sort_values(by='area').reset_index()
    gdf = gdf.rename(columns={'index': 'id'})
    small_polygons = gdf[gdf["area"] < threshold]
    big_polygons = gdf[gdf["area"] >= threshold]
    print(f'Small polygons: {small_polygons.shape[0]} \n Big polygons: {big_polygons.shape[0]}')
    if small_polygons.shape[0] > 0:
        # Iterate over each small polygon and find the ID of the closest big polygon
        for index, big_polygon in big_polygons.iterrows():
            b_poly = big_polygon.geometry
            near_by_polygon_loc = small_polygons.touches(b_poly)
            nearby_small_polygons = small_polygons[near_by_polygon_loc]
            if nearby_small_polygons.shape[0] > 0:
                nearby_small_polygons_geom = MultiPolygon(list(nearby_small_polygons.geometry))
                polygons = [big_polygon.geometry, nearby_small_polygons_geom]
                dissolved_poly = cascaded_union(polygons)
                big_polygons.loc[big_polygons.index == index, 'geometry'] = dissolved_poly
                small_polygons = small_polygons[~near_by_polygon_loc]
    return big_polygons, small_polygons


def merge_small_polygons_oldv(dataframe, threshold=0.3):
    gdf = dataframe.reset_index()
    gdf = gdf.rename(columns={'index': 'id'})
    small_polygons = gdf[gdf["area"] < threshold]
    big_polygons = gdf[gdf["area"] >= threshold]
    print(f'Small polygons: {small_polygons.shape[0]} \n Big polygons: {big_polygons.shape[0]}')

    # Create a spatial index for the big polygons
    big_polygons_sindex = big_polygons.sindex

    # Iterate over each small polygon and find the ID of the closest big polygon
    for index, small_polygon in small_polygons.iterrows():
        # Get the bounds of the small polygon
        s_poly = small_polygon.geometry
        # Query the spatial index of the big polygons to get the IDs of the nearby big polygons
        nearby_big_polygon_ids = list(big_polygons_sindex.intersection(s_poly.bounds))

        if len(nearby_big_polygon_ids) > 0:
            # Filter the big polygons DataFrame to only include the nearby big polygons
            nearby_big_polygons = big_polygons.iloc[nearby_big_polygon_ids]
            nearby_big_polygons['distance'] = nearby_big_polygons.geometry.distance(small_polygon.geometry)
            nearby_big_polygons = nearby_big_polygons.sort_values(by=['distance', 'area'], ascending=[True, False])
            id_of_big_polygon = nearby_big_polygons['id'].iloc[0]

            # dissolve both polygons
            big_polygon = big_polygons.loc[big_polygons['id'] == id_of_big_polygon, 'geometry']
            small_polygon_ = small_polygons.loc[small_polygons['id'] == small_polygon.id, 'geometry']
            polygons = [big_polygon, small_polygon_]
            x = cascaded_union(polygons)
            big_polygons.loc[big_polygons['id'] == id_of_big_polygon, 'geometry'] = x
        else:
            print("Something")  # fix-me
    return big_polygons


def create_label_id(shapefile, label_atr, dataset_name='nacala'):
    if dataset_name == 'nacala':
        roof_types = ['metal_sheet', 'thatch', 'asbestos', 'concrete', 'coconut:leaves', 'roof_tiles']
        label_ids = [1, 2, 3, 4, 2, 1]
        gdf = gpd.read_file(shapefile)
        gdf['label_id'] = gdf[label_atr]
        gdf['label_id'] = gdf['label_id'].replace(roof_types, label_ids)
        gdf["label_id"].fillna(5, inplace=True)
        gdf['label_id'] = gdf['label_id'].astype('int')
        new_file = shapefile
        gdf.to_file(new_file)
    else:
        print("change dataset")


def create_weight_polygon(shapefile, key, out_folder, buffer_dist=0.5):
    print(f'Creating weght polygons for {shapefile}')
    # assuming the coordinates are in meters
    dataframe = gpd.read_file(shapefile).set_crs(4326)
    buffer_gdf = dataframe.to_crs(3857)
    # buffer_gdf = dataframe.copy()
    buffer_gdf['geometry'] = buffer_gdf['geometry'].buffer(buffer_dist)
    buffer_gdf = buffer_gdf.to_crs(4326)

    # get intersection between every two polygons and create as a polygon
    new_index = 0
    new_polygon_list = []
    for index, row in buffer_gdf.iterrows():
        current_row = buffer_gdf.index.isin([index])
        temp = buffer_gdf[~current_row]
        intersecting_polygons_loc = temp.intersects(row.geometry)
        intersecting_polygons = temp[intersecting_polygons_loc]
        if intersecting_polygons.shape[0] > 1:
            for index2, row2 in intersecting_polygons.iterrows():
                intersection = row.geometry.intersection(row2.geometry)
                new_polygon_list.append([new_index, intersection])
                new_index += 1
    overlap_df = gpd.GeoDataFrame(new_polygon_list, columns=['_id', 'geometry']).set_crs(4326, inplace=True)
    # clip main buildings with ovelap polygons
    new_buildings = overlay(dataframe, overlap_df, how="difference")
    new_labels_file = os.path.join(out_folder, key + 'new_labels.shp')
    weights_file = os.path.join(out_folder, key + 'label_weights.shp')
    overlap_df.to_file(weights_file)
    new_buildings.to_file(new_labels_file)
    print(overlap_df.crs, new_buildings.crs)
    print(f'Done with creating weght polygons for {shapefile}')
    return new_labels_file, weights_file