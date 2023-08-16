import random
import warnings
import collections
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from tqdm import tqdm
# geometry classes and methods from shapely
from shapely.geometry import LineString, Polygon, MultiPolygon, MultiPoint, Point
from shapely.ops import polygonize
# voronoi basics from scipy
from scipy.spatial import Voronoi
import warnings


def allocate(distribution_table, demands):
    # demands = hours x regions
    # distrib = nodes x regions
    # allocat = hours x nodes
    demands = demands.fillna(0)
    for i, region in enumerate(distribution_table.columns):
        if region not in demands.columns:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                demands[region] = [0]*len(demands)
        if i == 0:
            m = np.outer(demands[region], distribution_table[region])
        else:
            m += np.outer(demands[region], distribution_table[region])

    allocated = pd.DataFrame(data=m, index=demands.index, columns=distribution_table.index)

    if not 0.99 < demands.sum().sum()/allocated.sum().sum() < 1.01:
        warnings.warn('WARNING: Allocation Error!')

    return allocated


# --- VORONOI DIAGRAM TOOL --------------------------------------------------------------------------------------------

# helper methods

def points_to_coordinates(points):
    """turn a list of points into an array of coordinates"""
    coordinates = np.zeros((len(points), 2))
    for i, point in enumerate(points):
        coordinates[i] = np.array([point.x, point.y])
    return coordinates


def mirror_x(point, x):  # this could fail with negative coordinates!
    """mirror coordinates parallel to y-axis at a given x"""
    # get the difference between point(x) and x and add that to x
    difference = point[0] - x
    point[0] = x - difference
    return point


def mirror_y(point, y):  # this could fail with negative coordinates!
    """mirror coordinates parallel to x-axis at a given y"""
    # get the difference between point(x) and x and add that to x
    difference = point[1] - y
    point[1] = y - difference
    return point


def mirror_coordinates(coordinates, buffer):
    """takes a list of coordinates and mirrors them to the right, the left, the top and the bottom.
    Necessary to make sure the voronoi polygon is bound correctly i.e. outside polygons are not infinite"""
    # creating the convex hull around nodes by connecting its outermost nodes and adding the buffer
    convex_hull = MultiPoint([Point(i) for i in coordinates]).convex_hull.buffer(buffer)

    # find the leftmost, rightmost, topmost, lowermost points of the hull - these will be used, as mirror axes
    hull_coordinates = [c for c in convex_hull.exterior.coords]
    top = hull_coordinates[0][1]
    bottom = hull_coordinates[0][1]
    left = hull_coordinates[0][0]
    right = hull_coordinates[0][0]
    for point in hull_coordinates:
        if point[0] < left:
            left = point[0]
        elif point[0] > right:
            right = point[0]
        if point[1] < bottom:
            bottom = point[1]
        elif point[1] > top:
            top = point[1]

    # now we have the left, right, top and bottom mirror axes and can mirror all coordinates
    top_coordinates = [mirror_y(point, top) for point in coordinates.copy()]
    left_coordinates = [mirror_x(point, left) for point in coordinates.copy()]
    right_coordinates = [mirror_x(point, right) for point in coordinates.copy()]
    bottom_coordinates = [mirror_y(point, bottom) for point in coordinates.copy()]
    new_coordinates = np.vstack((coordinates, left_coordinates, top_coordinates, right_coordinates, bottom_coordinates))
    return new_coordinates


# main functions

def create_voronoi_diagram(nodes_gdf, buffer):
    """
    - get voronoi diagram from coordinates (nodes_gdf) mirrored along 4 axes around a buffer using scipy.spatial Voronoi
    - turn Voronoi object into polygons
    """
    #print('\tCreating Voronoi diagram from node coordinates...')
    coordinates = points_to_coordinates(list(nodes_gdf.geometry))
    mirrored_coordinates = mirror_coordinates(coordinates, buffer)
    vor = Voronoi(mirrored_coordinates)
    #print('\tCreating Voronoi cells from the diagram...')
    # turn Voronoi cells into shapely LineStrings and then Polygons
    lines = [LineString(vor.vertices[line]) for line in vor.ridge_vertices if -1 not in line]
    result = [poly for poly in polygonize(lines)]
    polys_gdf = gpd.GeoDataFrame({'geometry': result})
    polys_gdf.crs = nodes_gdf.crs
    return polys_gdf


def map_polys_to_nodes(polygons_gdf, points_gdf, point_id):
    """create a dictionary of node_ids to polygons"""
    #print('\tMapping node ids to polygons...')
    matches_dict = {}
    unmatched_nodes = []
    # ideas to make this faster:
    # - get rid of dataframe and use dict.
    # - keep track of mapped points (pop from list of points)
    for index, row in points_gdf.iterrows():
        point = row['geometry']
        node = row[point_id]
        for polygon in polygons_gdf.geometry:
            if isinstance(polygon, MultiPolygon):
                for poly in polygon:
                    if poly.contains(point):
                        matches_dict.update({node: polygon})
                        break
                        # if one node inside the polygon is found we can leave the loop because there can never be two
                        # --> definition of voronoi diagram
            elif isinstance(polygon, Polygon):
                if polygon.contains(point):
                    matches_dict.update({node: polygon})
                    break  # if one node is inside the polygon we can leave the loop
        else:  # for else --> if point is not in any of the polygons put it onto another list
            unmatched_nodes.append(node)
    if unmatched_nodes:
        print(len(unmatched_nodes), 'node(s) could not be assigned to a polygon:')
        print(unmatched_nodes)
    matches_gdf = gpd.GeoDataFrame({'node_id': list(matches_dict.keys()), 'geometry': list(matches_dict.values())})
    matches_gdf.crs = polygons_gdf.crs
    return matches_gdf


def find_prop_of_area(voro_gdf, regions_gdf, region_id, point_id, common_projection=3035):
    """overlay regions and polygons to calculate share of each regions area covered by each polygon.
    The resulting values will be summed and can be used as allocation keys."""
    #print("\tCalculating the area share of each polygon for all regions...")
    # EPSG 3035 is the Lambert Azimuthal Equal Area projection that covers all of Europe without distortion
    voro_gdf.to_crs(epsg=common_projection, inplace=True)
    regions_gdf.to_crs(epsg=common_projection, inplace=True)
    # intersect regions and polygons
    intersect_gdf = gpd.overlay(voro_gdf, regions_gdf[['geometry', region_id]], how='intersection')
    intersect_gdf.crs = f'epsg:{common_projection}'
    # find intersection area
    intersect_gdf['area'] = intersect_gdf['geometry'].area
    # build result df
    res_df = pd.DataFrame(columns=intersect_gdf[region_id].unique(), index=intersect_gdf[point_id].unique())
    for regions_gdf in res_df.columns:
        # select area data
        mask = intersect_gdf[region_id] == regions_gdf
        tot_area = intersect_gdf.loc[mask, 'area'].sum()
        rel_nodes = intersect_gdf.loc[mask, point_id].tolist()
        for node in rel_nodes:
            res_df.loc[node, regions_gdf] = intersect_gdf.loc[mask].set_index(point_id).loc[node, 'area'] / tot_area
    res_df.fillna(0., inplace=True)
    return res_df, intersect_gdf


def _random_test(gdf1, gdf2, point_id, show_plots=True):
    """test whether mapping of nodes to polygons is correct by displaying a random node and its surrounding polygon"""
    if show_plots:
        #print(
        #    '\tConfirming correct mapping of nodes to polygons with a random test (node should lie within polygon).')
        node_id_list = list(gdf2[point_id])
        random_id = random.choice(node_id_list)
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        gdf1[gdf1[point_id] == random_id].plot(ax=ax, facecolor='none', edgecolor='black')
        gdf2[gdf2[point_id] == random_id].plot(ax=ax, marker='o', color='red', markersize=10)
        plt.show()


def show(header, gdf1, gdf2):
    """plot GeoDataFrame geometries, gdf1 = polygons, gdf2 = points"""
    fig, ax = plt.subplots(figsize=[12, 9])
    ax.set_title(label=header)
    ax.set_aspect('equal')
    gdf1.plot(ax=ax, color='white', edgecolor='black')
    gdf2.plot(ax=ax, marker='o', color='red', markersize=1)
    plt.show()


def check_plausibility(results_df, voro_gdf, nodes_gdf, regions_gdf, intersect_gdf,
                       region_id, point_id, show_plots=True):
    """Method to combine all plausibility checks"""
    #print('\tTesting results for plausibility...')
    #print('\tNumber of nodes:', len(nodes_gdf.index))
    #print('\tNumber of polygons:', len(voro_gdf.index))

    # nodes_gdf.to_crs(epsg=common_projection, inplace=True)
    # _random_test(voro_gdf, nodes_gdf, point_id)
    if show_plots:
        fig, ax = plt.subplots(figsize=[12, 9])
        intersect_gdf.plot(ax=ax, color='lightgreen', alpha=0.4, edgecolor='darkgreen')
        plt.title('Intersection of regions and voronoi cells (darker shade indicates duplicated nodes)')
        ax.set_aspect('equal')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.show()
    # print('Areas with darker shade are duplicated...')

    lost = results_df.sum(axis=0)[results_df.sum(axis=0).round(2) != 1]
    if len(lost) > 0:
        warnings.warn('There are {len(lost)} regions, where the area share does not add up to 1. '
                      'For each region the shares of each polygon should add up to 1, '
                      'as 100% of each region has to be covered by the polygons.')
    if show_plots:
        random_rs = random.choice(list(results_df.columns))
        random_rs_polys = {}
        for index in results_df.index:
            if results_df.loc[index, random_rs] != 0:
                random_rs_polys.update({index: results_df.loc[index, random_rs]})
        # we now have a dict for one region of node id : area share
        # make it sorted by node id and then also sort the result df by node id
        random_rs_polys = collections.OrderedDict(sorted(random_rs_polys.items()))
        # print('Node IDs of intersecting polygons:\n', list(random_rs_polys.keys()))
        # print('Sum of their area shares:', sum(random_rs_polys.values()))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            polys_plot_gdf = voro_gdf.sort_values(point_id)[voro_gdf[point_id].isin(list(random_rs_polys.keys()))]
        centroids = [poly.centroid for poly in list(polys_plot_gdf.geometry)]  # get the centroids
        x = [c.x for c in centroids]
        y = [c.y for c in centroids]
        words = [(p * 100).round(1) for p in list(random_rs_polys.values())]
        fig, ax = plt.subplots(figsize=[12, 9])
        ax.set_title(label='Random region with intersecting polygons and their area share')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_aspect('equal')

        # plot the region
        regions_gdf.loc[regions_gdf[region_id] == random_rs].plot(ax=ax, color='red', alpha=0.2, edgecolor='black')
        polys_plot_gdf.plot(ax=ax, facecolor='none', edgecolor='red')  # plot the polys
        for i, word in enumerate(words):
            plt.text(x[i], y[i], str(word) + '%', fontsize=9)
        plt.show()
