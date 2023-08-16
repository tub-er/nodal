
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import warnings
import matplotlib.pyplot as plt
from functools import reduce
from os.path import exists
from shapely.geometry import Point, Polygon, MultiPolygon, LineString
from math import radians, cos, sin, asin, sqrt
from scipy import spatial
from scalegrid_connection import get_geodata
from voronoi import show, create_voronoi_diagram, map_polys_to_nodes, find_prop_of_area, check_plausibility, \
        allocate
import xml.etree.ElementTree as ET


point_weight = pd.read_csv('input/DE_point_weight.csv', sep=';')
point_weight.set_index('node_id', inplace=True)

if exists('input/admin_regions.pkl'):
    with open('input/admin_regions.pkl', 'rb') as file:
        regions = pickle.load(file)
else:
    regions = get_geodata(db='geodata', coll='administrative_borders',
                          query={"properties.type": "KRS", "properties.year": 2020, "properties.EWZ": {'$gt': 0}})
    with open('input/admin_regions.pkl', 'wb') as file:
        pickle.dump(regions, file)


def xml_input_nodes(filename):
    nodes, elements = read_topology_xml(filename)
    nodes = nodes[nodes.name.str.contains('GS')]
    supply = nodes[nodes.supply == 1].name.tolist()
    nd_dct = {name: (x, y) for name, x, y in zip(nodes['name'], nodes['x'], nodes['y'])}
    return nd_dct, supply


def read_topology_xml(name):
    nodes = {}
    elements = {}

    root = ET.parse(name + '.xml').getroot()
    for child in root:
        if child.tag == 'NODES':
            for elm in child:
                nodes[elm.attrib['id']] = {'name': elm.attrib['name'],
                                           'x': float(elm.attrib['x']),
                                           'y': float(elm.attrib['y']),
                                           'supply': (1 if 'supply' in elm.attrib else 0)}
        elif child.tag == 'ELEMENTS':
            for elm in child:
                if elm.attrib['type'] == 'pipe':
                    length = float(elm.attrib['length'])
                else:
                    length = 0.01
                elements[elm.attrib['id']] = {'name': elm.attrib['name'],
                                              'node0': elm[0].attrib['id'],
                                              'node1': elm[1].attrib['id'],
                                              'type': elm.attrib['type'],
                                              'diameter': float(elm.attrib['diameter']),
                                              'length': length}
        else:
            pass

    nodes = pd.DataFrame(nodes).T
    elements = pd.DataFrame(elements).T
    
    return nodes, elements


def map_coords_to_regions(data):
    mapping = {}
    for _, row in regions.iterrows():
        poly = row['geometry']
        region = row['NUTS']
        region_coords = []
        for coord in data.columns:
            if poly.geom_type == 'MultiPolygon':
                for p in poly.geoms:
                    if p.contains(Point(coord)):
                        region_coords.append(coord)
                        break
            elif poly.contains(Point(coord)):
                region_coords.append(coord)
        if region_coords:
            mapping.update({region: region_coords})

    mapping_reversed = {}
    for region, coords in mapping.items():
        for c in coords:
            mapping_reversed.update({c: region})
    regional_data = data.rename(columns=mapping_reversed).groupby(axis=1, level=0).sum()
    return regional_data


def _allocate_nearest(nodes, data, max_allowable_distance=20):

    if not nodes:
        return pd.DataFrame()
    
    def haversine(lat1, lon1, lat2, lon2):
        """ Calculate the great circle distance between two points
        on the earth (specified in decimal degrees) """
        # convert decimal degrees to radians
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        # haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        r = 6371  # Radius of earth in kilometers. Use 3956 for miles
        return c * r

    def _find_nearest(point, points):
        opts = spatial.KDTree(points)  # k-d-tree of coords for nearest search
        _, ind = opts.query(point)
        distance = haversine(point[0], point[1], points[ind][0], points[ind][1])  # km
        return points[ind], distance

    coords = {nodes[node]: node for node in nodes}
    coord_list = list(coords.keys())  # coords of nodes in the topology

    closest = {}
    for p in data.columns:
        closest_coord, dist = _find_nearest(p, coord_list)
        closest_node = coords[closest_coord]
        if dist > max_allowable_distance:
            print(f'\tNode Far Away Warning: {closest_node=},  dist={round(dist,1)} [km]')
        closest.update({p: closest_node})
    return data.rename(columns=closest).groupby(axis=1, level=0).sum()


def _allocate_voronoi(dso, nodes):

    regions_temp = regions.copy(deep=True)
    seeds = gpd.GeoDataFrame({'node_id': nodes.keys(), 'geometry': [Point(x) for x in nodes.values()]}, crs="EPSG:4326")
    # allocate demands based on voronoi approach for regions without nodes
    voro = create_voronoi_diagram(seeds, buffer=10)
    voro = map_polys_to_nodes(voro, seeds, point_id='node_id')
    # check whether polygons and nodes are overlapping as expected
    show('Polygons and corresponding nodes', voro, seeds)
    distribution_table, intersection = find_prop_of_area(voro, regions_temp, region_id='NUTS', point_id='node_id',
                                                         common_projection=3035)
    check_plausibility(distribution_table, voro, seeds, regions_temp, intersection,
                       show_plots=False, region_id='NUTS', point_id='node_id')
    allocated = allocate(distribution_table, dso)

    return allocated


def _allocate_weight(data, nodes):
    # allocate demands based on weight table for regions with nodes
    def _map_nodes_regions(nodes):
        mapping = {}

        # fig, ax = plt.subplots(figsize=(10,15))
        # regions.plot(ax=ax, facecolor=None)
        # gpd.GeoDataFrame(index=nodes.keys(), geometry=[Point(x) for x in nodes.values()]).plot(ax=ax, color='red')
        # plt.show()

        for _, row in regions.iterrows():
            poly = row['geometry']
            region = row['NUTS']
            region_nodes = []
            for node in nodes:
                if poly.geom_type == 'MultiPolygon':
                    for p in poly.geoms:
                        if p.contains(Point(nodes[node])):
                            region_nodes.append(node)
                            break
                elif poly.contains(Point(nodes[node])):
                    region_nodes.append(node)
            if region_nodes:
                mapping.update({region: region_nodes})
        return mapping

    mapping = _map_nodes_regions(nodes)
    result = {}
    remainder = []
    for region in data.columns:
        # check which nodes are in the region
        region_nodes = mapping.get(region, [])
        # if there are none keep the data for voronoi approach
        if not region_nodes:
            remainder.append(data[region])
            continue
        weights = {}
        for node in region_nodes:
            if node in point_weight.index:
                weights.update({node: point_weight.loc[node, 'weight']})
            else:
                weights.update({node: 0})
        try:
            n_weights = {node: w / sum(weights.values()) for node, w in weights.items()}
        except ZeroDivisionError:
            remainder.append(data[region])
            continue
        for node in region_nodes:
            result.update({node: list(n_weights[node] * data[region])})
    return pd.DataFrame(result), pd.concat(remainder, axis=1)

