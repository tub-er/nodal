
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
    
    # filter out compressor inlet nodes
    # check if this works:
    #nodes = nodes[~[elements[elements.type=='compressor station']['node0'].values]]
    
    return nodes, elements


def map_coords_to_regions(data):
    # when we get coordinate demands, we can allocate them to regions and
    # use the normal allocation methods
    # this helps with simulation because it is less likely large demands
    # will be allocated to small nodes just because they are closest.
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
    # change the data to NUTS3 column format using the mapping
    regional_data = data.rename(columns=mapping_reversed).groupby(axis=1, level=0).sum()
    return regional_data


def _allocate_nearest(nodes, data):

    if not nodes:
        return pd.DataFrame()
    # allocate demands based on coordinates
    # nodes is a dictionary with nodes pointing to coordinates (from the topology)
    # data is a dataframe with coordinate tuples in the columns and timesteps in the rows
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
        if dist > 20:
            pass
            # print(f'\tNode Far Away Warning: {closest_node=},  dist={round(dist,1)} [km]')
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


def allocate_industry(scenario, year, gas_type):
    # allocate to nodes based on coordinates
    # get topology
    if gas_type == 'H2':
        terminal = pd.read_csv(f'input/{scenario}_topo/terminal/terminal_csv/terminal{year}.csv')
        # for industry type nodes
        nodelist = list(terminal[terminal.type == 'IND'].iloc[:, 1])
        nodelist = [i for i in nodelist if i not in ['GS-0770', 'GS-0643', 'GS-0768']]  # remove manually some nodes
        nodes, supply = xml_input_nodes(f'input/{scenario}_topo/topo_h2/TE_{year}')
        nodes_ind = {k: v for k, v in nodes.items() if k in nodelist}
    else:
        nodes, supply = xml_input_nodes(f'input/{scenario}_topo/topo_c4/TE_{year}')
        de_nodes = pd.read_csv('input/DE2022_nodes.csv')
        de_nodes_ind = de_nodes[de_nodes['node_type'].isin(['IND'])].node_name.to_list()
        nodes_ind = {k: v for k, v in nodes.items() if k in de_nodes_ind}

    tso = pd.read_excel(f'output/{scenario}_{year}_{gas_type}_industrial_tso.xlsx')
    tso.drop('Unnamed: 0', axis=1, errors='ignore', inplace=True)
    tso.rename(columns={'DEB16': 'DEB1C', 'DEB19': 'DEB1D'}, inplace=True)

    dso = pd.read_excel(f'output/{scenario}_{year}_{gas_type}_industrial_dso.xlsx')
    dso.rename(columns={'DEB16': 'DEB1C', 'DEB19': 'DEB1D'}, inplace=True)
    dso.drop('Unnamed: 0', axis=1, errors='ignore', inplace=True)

    #if scenario != 'T45-Strom' or gas_type == 'CH4':
    #    tso = tso / 1000
    #    dso = dso / 1000

    allocated_tso, remainder = _allocate_weight(tso, nodes_ind)
    unallocated = dso.add(remainder, fill_value=0)
    print('\tDSO demands:', round(unallocated.sum().sum()/1_000_000, 1))
    print('\tTSO demands:', round(allocated_tso.sum().sum()/1_000_000, 1))

    return unallocated, allocated_tso


def allocate_dso(year, scenario, gas_type, ind):

    # load all the tables
    def load_file(filename):
        if exists(filename):
            print(f'\tloading {filename}...')
            data = pd.read_excel(filename)
            data.drop(['Unnamed: 0', 'date_time'], axis=1, errors='ignore', inplace=True)
            data.rename(columns={'DEB16': 'DEB1C', 'DEB19': 'DEB1D'}, inplace=True)
            data = data / 1000  # convert MWh
            print(f'\t{round(data.sum().sum()/1000_000, 1)} TWh')
            return data
        else:
            print(f'\t{filename} does not exist.')
            return pd.DataFrame()

    if exists(f'output/dso_demands_total_{gas_type}_{scenario}_{year}.pkl'):
        with open(f'output/dso_demands_total_{gas_type}_{scenario}_{year}.pkl', 'rb') as file:
            dso = pickle.load(file)
    else:
        appl = load_file(f'output/{scenario}_{year}_{gas_type}_appliances.xlsx')
        heat_ph = load_file(f'output/{scenario}_{year}_{gas_type}_heating_hh.xlsx')
        heat_cts = load_file(f'output/{scenario}_{year}_{gas_type}_heating_cts.xlsx')
        traffic = load_file(f'output/{scenario}_{year}_{gas_type}_traffic.xlsx')
        # sum up all the tables
        dso = reduce(lambda a, b: a.add(b, fill_value=0), [ind, appl, heat_ph, heat_cts, traffic])
        with open(f'output/dso_demands_total_{gas_type}_{scenario}_{year}.pkl', 'wb') as file:
            pickle.dump(dso, file)

    # for dso nodes
    if gas_type == 'H2':
        terminal = pd.read_csv(f'input/{scenario}_topo/terminal/terminal_csv/terminal{year}.csv')
        nodes, supply = xml_input_nodes(f'input/{scenario}_topo/topo_h2/TE_{year}')
        nodelist = list(terminal[~terminal.type.isin(['IND', 'IC', 'STO', 'LNG'])].iloc[:, 1])
        nodelist = [i for i in nodelist if i not in ['GS-0770', 'GS-0643', 'GS-0768', 'GS-0088', 'GS-0086']]  # remove manually some nodes
        nodes_dso = {k: v for k, v in nodes.items() if k in nodelist}
    else:
        nodes, supply = xml_input_nodes(f'input/{scenario}_topo/topo_c4/TE_{year}')
        de_nodes = pd.read_csv('input/DE2022_nodes.csv')
        de_nodes_dso = de_nodes[de_nodes['node_type'].isin(['DSO'])].node_name.to_list()
        nodes_dso = {k: v for k, v in nodes.items() if k in de_nodes_dso}

    print('\tDSO demands before allocation:', round(dso.sum().sum()/1_000_000, 1))
    allocated_weight, remainder = _allocate_weight(dso, nodes_dso)
    print('\tDSO demands after allocation 1st step:', round((allocated_weight.sum().sum() + remainder.sum().sum())/1_000_000, 1))

    # WE ARE NOT ALLOCATING ANY DEMANDS FROM REGIONS WITHOUT NODES
    # WELL ACTUALLY, IF TOO MUCH DEMAND IS LOST (MORE THAN 10%) WE WILL JUST REMOVE AS MUCH AS WAS LOST DURING REMOVAL OF IMPORTS

    # print('\tDSO demands after allocation 2nd step: (removed regions without nodes)', round(allocated_weight.sum().sum()/1_000_000, 1))
    # return allocated_weight
    allocated_dso = _allocate_voronoi(remainder, nodes_dso)
    result = allocated_dso.add(allocated_weight, fill_value=0)
    print('\tDSO demands after allocation 2nd step:', round(result.sum().sum()/1_000_000, 1))
    return result  # mw

def pp_elec_to_nuts_h2(scenarios):
    pp = []
    ee = []
    for scenario in scenarios:
        for year in [2025, 2030, 2035, 2040, 2045]:

            pp_allocated, ee_allocated = allocate_pp_elec_h2(year, scenario, nuts3_only=True)

            pp_col = pp_allocated.sum()
            pp_col.index = pp_col.index.map(str)
            if pp_col.sum() != 0:
                pp_col.name = f'{scenario}_{year}'
                pp_col = pp_col.groupby(axis=0, level=0).sum()
                pp.append(pp_col)

            ee_col = ee_allocated.sum()
            ee_col.index = ee_col.index.map(str)
            if ee_col.sum() != 0:
                ee_col.name = f'{scenario}_{year}'
                ee_col = ee_col.groupby(axis=0, level=0).sum()
                ee.append(ee_col)

    pp_df = pd.concat(pp, axis=1)
    ee_df = pd.concat(ee, axis=1)
    pp_df.to_excel('pp_per_coord.xlsx')
    ee_df.to_excel('ee_per_coord.xlsx')


def allocate_pp_elec_h2(year, scenario, nuts3_only=False):
    # actual electrolysis yearly in these scenarios:
    actual_elec = {'T45-RedEff': {2025: 119_000,
                                  2030: 23_836_000,
                                  2035: 60_164_000,
                                  2040: 124_443_000,
                                  2045: 129_603_000},
                   'T45-RedGas': {2025: 136_000,
                                  2030: 23_255_000,
                                  2035: 72_335_000,
                                  2040: 142_283_000,
                                  2045: 173_765_000}
                   }

    actual_pwpl = {'T45-RedGas': {2025: 868,
                                  2030: 2173872,
                                  2035: 15187831,
                                  2040: 93699721,
                                  2045: 102130732},
                   'T45-RedEff': {2025: 1290,
                                  2030: 3219205,
                                  2035: 26667996,
                                  2040: 132581816,
                                  2045: 142319023}
                   }

    # get the Verteilschlüssel
    temp = scenario
    if scenario in ['T45-RedGas', 'T45-RedEff']:
        multiplicator_e = actual_elec[scenario][year]
        multiplicator_p = actual_pwpl[scenario][year]
        scenario = 'T45-Strom'
    try:
        ratio_keys = pd.read_excel(f'input/Verteilschlüssel_{scenario}_TUB.xlsx', sheet_name=str(year))
    except FileNotFoundError:
        ratio_keys = pd.read_excel(f'input/H2-Verteilschluessel_{scenario}.xlsx', sheet_name=str(year))
    ratio_keys['point'] = list(zip(ratio_keys.Longitude, ratio_keys.Latitude))

    # import hourly demand and production
    electrolysis = pd.read_csv(f'data_tables/electrolysis_{scenario}_{year}.csv')
    power_plants = pd.read_csv(f'data_tables/tpp_h2_{scenario}_{year}.csv')

    # adjust the yearly amount of the hourly T45-Strom data for RedEff and RedGas
    if temp != scenario:
        new_val_e = electrolysis.value / abs(electrolysis.value.sum()) * multiplicator_e
        electrolysis.value = new_val_e
        new_val_p = power_plants.value / abs(power_plants.value.sum()) * multiplicator_p
        power_plants.value = new_val_p

    # reset scenario
    scenario = temp

    print('PP before regional loss:', round(power_plants.value.sum() / 1_000_000, 1))
    print('ELEC before regional loss:', round(electrolysis.value.sum() / 1_000_000, 1))

    # check which enertile zones have nodes
    terminal = pd.read_csv(f'input/{scenario}_topo/terminal/terminal_csv/terminal{year}.csv')
    nodes, supply = xml_input_nodes(f'input/{scenario}_topo/topo_h2/TE_{year}')

    pp_nodes = list(terminal[terminal.type.str.contains('PP')].iloc[:, 1])
    ee_nodes = list(terminal[terminal.type.str.contains('EE')].iloc[:, 1])

    pp_coords = [nodes[n] for n in pp_nodes if n in nodes]
    ee_coords = [nodes[n] for n in ee_nodes if n in nodes]
    # get enertile zone
    # get a boundary of all the pp/ee locations and check if there are nodes within
    # since we dont have any geodata for enertile zones - not needed anymore
    #boundaries = {}
    pp_zones = ['DE_1', 'DE_2', 'DE_3', 'DE_4', 'DE_5', 'DE_6']
    ee_zones = ['DE_1', 'DE_2', 'DE_3', 'DE_4', 'DE_5', 'DE_6']

    #for region in ['DE_1', 'DE_2', 'DE_3', 'DE_4', 'DE_5', 'DE_6']:
    #    region_keys = ratio_keys[ratio_keys.Region == region]
    #    points = list(region_keys['point'])
    #    # create a polygon around all the locations
    #    # this does not work if we only have two points...
    #    try:
    #        enertile_hull = Polygon(points).convex_hull.buffer(0.7)
    #    except ValueError:
    #
    #    # check if any points are within the region
    #    for p in pp_coords:
    #        if Point(p).within(enertile_hull):
    #            pp_zones.append(region)
    #            break
    #    for p in ee_coords:
    #        if Point(p).within(enertile_hull):
    #            ee_zones.append(region)
    #            break
    #    boundaries.update({region: enertile_hull})
    
    #fig, ax = plt.subplots(figsize=(9,9))
    #gdf = gpd.GeoDataFrame(geometry=list(boundaries.values()))
    #gdf.plot(ax=ax, edgecolor='blue', facecolor='None', linewidth=2)
    #with warnings.catch_warnings():
    #    warnings.simplefilter("ignore")
    #    gdf1 = gpd.GeoDataFrame(geometry=[Point(i) for i in pp_coords])
    #    gdf1.plot(ax=ax, color='red')
    #    gdf2 = gpd.GeoDataFrame(geometry=[Point(i) for i in ee_coords])
    #    gdf2.plot(ax=ax, color='green')
    #    plt.title('Using these approximations of enertile zones based on electrolyser / power plant location coordinates\nif there are any nodes (PP red/EE green) within these polygons, we can allocate the zones pp/el there')
    #    plt.show()

    results_tpp = []
    results_pth = []

    # apply distribution keys for each Enertile zone
    for region in pp_zones:
        tpp = power_plants[power_plants.region == region][['value', 'hour']].groupby('hour').sum()
        try:
            ratio = ratio_keys[ratio_keys.Region == region].set_index('point')['relativer Schlüssel']
        except KeyError:
            ratio = ratio_keys[ratio_keys.Region == region].set_index('point')['rel. Schlüssel']
        results_tpp.append(pd.DataFrame(index=tpp.index, columns=ratio.index, data=np.outer(tpp, ratio)))
    try:
        tpp_df = pd.concat(results_tpp, axis=1) * (-1)
    except ValueError:
        tpp_df = pd.DataFrame()
    print('TPP:', round(tpp_df.sum().sum() / 1_000_000, 1))

    for region in ee_zones:
        pth = electrolysis[electrolysis.region == region][['value', 'hour']].groupby('hour').sum()
        try:
            ratio = ratio_keys[ratio_keys.Region == region].set_index('point')['relativer Schlüssel']
        except KeyError:
            ratio = ratio_keys[ratio_keys.Region == region].set_index('point')['rel. Schlüssel']
        results_pth.append(pd.DataFrame(index=pth.index, columns=ratio.index, data=np.outer(pth, ratio)))

    try:
        pth_df = pd.concat(results_pth, axis=1) * (-1)
    except ValueError:
        pth_df = pd.DataFrame()
    print('ELEC:', round(pth_df.sum().sum() / 1_000_000, 1))

    # if sum is different in T45-RedEff and RedGas, change here

    #fig, ax = plt.subplots(figsize=(15,10))
    #tpp_df.sum(axis=1).plot(ax=ax, label='Power Plants')
    #pth_df.sum(axis=1).plot(ax=ax, label='Electrolysis')
    #plt.title(f'H2 PowerPlant demands/electrolysis {scenario} {year}')
    #plt.legend()
    #plt.show()
    if nuts3_only:
        return tpp_df, pth_df

    nodes_pp = {n: nodes[n] for n in pp_nodes if n in nodes}
    nodes_ee = {n: nodes[n] for n in ee_nodes if n in nodes}

    pp_allocated = _allocate_nearest(data=tpp_df, nodes=nodes_pp)
    ee_allocated = _allocate_nearest(data=pth_df, nodes=nodes_ee)

    return pp_allocated, ee_allocated

# we have regional ratio and efficiency as well as hourly data
#def allocate_tpp_electrolysis_h2_old(year, scenario):
#
#    # import H2 distribution keys
#    ratio_keys = pd.read_excel(f'input/H2-Verteilschluessel_{scenario}_analog39847_20220921.xlsx', sheet_name=str(year))
#    ratio_keys['point'] = list(zip(ratio_keys.Longitude, ratio_keys.Latitude))
#
#    # import hourly demand and production
#    if scenario in ['T45-RedGas', 'T45-RedEff']:
#        electrolysis = pd.read_csv(f'data_tables/electrolysis_T45-Strom_{year}.csv')
#        power_plants = pd.read_csv(f'data_tables/tpp_h2_T45-Strom_{year}.csv')
#    else:
#        electrolysis = pd.read_csv(f'data_tables/electrolysis_{scenario}_{year}.csv')
#        power_plants = pd.read_csv(f'data_tables/tpp_h2_{scenario}_{year}.csv')
#
#    results_tpp = []
#    results_pth = []
#
#    # apply distribution keys for each Enertile zone
#    for region in ratio_keys.Region.unique():
#        tpp = power_plants[power_plants.region == region][['value', 'hour']].groupby('hour').sum()
#        pth = electrolysis[electrolysis.region == region][['value', 'hour']].groupby('hour').sum()
#        ratio = ratio_keys[ratio_keys.Region == region].set_index('point')['relativer Schlüssel']
#        results_tpp.append(pd.DataFrame(index=tpp.index, columns=ratio.index, data=np.outer(tpp, ratio)))
#        results_pth.append(pd.DataFrame(index=pth.index, columns=ratio.index, data=np.outer(pth, ratio)))
#    tpp_df = pd.concat(results_tpp, axis=1) * (-1)
#    print('TPP:', round(tpp_df.sum().sum()/1000_000, 1))
#
#    pth_df = pd.concat(results_pth, axis=1) * (-1)
#    print('ELEC:', round(pth_df.sum().sum()/1000_000, 1))
#
#    #fig, ax = plt.subplots(figsize=(15,10))
#    #tpp_df.sum(axis=1).plot(ax=ax, label='Power Plants')
#    #pth_df.sum(axis=1).plot(ax=ax, label='Electrolysis')
#    #plt.title(f'H2 PowerPlant demands/electrolysis {scenario} {year}')
#    #plt.legend()
#    #plt.show()
#
#    # allocate to nodes based on coordinates
#    terminal = pd.read_csv(f'input/{scenario}_topo/terminal/terminal_csv/terminal{year}.csv')
#    pp_nodelist = list(terminal[terminal.type.str.contains('PP|EE', regex=True)].iloc[:, 1])
#    pp_nodelist = [i for i in pp_nodelist if i not in ['GS-0770', 'GS-0643', 'GS-0768']]  # remove manually some nodes
#
#    # get topology
#    nodes, supply = read_simone_txt(f'input/{scenario}_topo/topo_h2/h2_{year}.txt')
#    nodes_pp = {k: v for k, v in nodes.items() if k in pp_nodelist}  # filter power plant nodes
#
#    allocated = _allocate_nearest(data=tpp_df.add(pth_df, fill_value=0), nodes=nodes_pp)
#
#    # get negative nodes and save them
#    supply_list = allocated.columns[(allocated < 0).any()].tolist()
#    with open(f'output/balanced/extra_h2_supply_nodes_{scenario}_{year}.txt', 'w') as file:
#        for node in supply_list:
#            file.write(node+'\n')
#
#    return allocated


def allocate_tpp_m(year, scenario):
    # allocate to nodes based on coordinates

    # get topology
    nodes, supply = xml_input_nodes(f'input/{scenario}_topo/topo_c4/TE_{year}')
    de_nodes = pd.read_csv('input/DE2022_nodes.csv')

    # the thought before was to use all the nodes available, in order to have less distance
    # de_nodes_pp = de_nodes[~de_nodes['node_type'].isin(['UGS', 'IC', 'interconnection'])].node_name.to_list()
    # However in order to have good simulation results we should instead use TPP nodes only as they allow higher demand
    de_nodes_pp = de_nodes[de_nodes['node_type'].isin(['TPP'])].node_name.to_list()
    # filter for PP nodes that are not in the h2 topology
    nodes_pp = {n: nodes[n] for n in nodes if n in de_nodes_pp}

    # get the data
    try:
        raw = pd.read_excel(f'data_tables/Schnittstelle_Gas_LFS3_{scenario}_{year}.xlsx')
    except FileNotFoundError:
        print(f'No file named Schnittstelle_Gas_LFS3_{scenario}_{year}.xlsx')
        return pd.DataFrame()
    raw['coords'] = list(zip(raw.Latitude, raw.Longitude))
    efficiency = raw['Wirkungsgrad'].tolist()
    raw.set_index('coords', inplace=True)
    data = raw[[str(i) for i in range(1,8760)]].copy()  # copy to avoid chained indexing with unpredictable behaviour

    # divide by efficiency to get methane demands
    for i, eff in enumerate(efficiency):
        try:
            data.iloc[i] = data.iloc[i] / eff
        except TypeError:
            data.iloc[i] = data.iloc[i] / float(eff.replace(',', '.'))

    # The method below is not used anymore because in created issues with Simone simulations
    allocated = _allocate_nearest(data=data.T, nodes=nodes_pp)
    # Instead we could do this:
    # regional_demands = map_coords_to_regions(data)
    # allocated_tso, remainder = _allocate_weight(regional_demands, nodes_pp)
    # allocated = allocated_tso.add(_allocate_voronoi(remainder, nodes_pp), fill_value=0)
    #
    #fig, ax = plt.subplots(figsize=(15, 10))
    #allocated.sum(axis=1).plot(ax=ax)
    #plt.title(f'CH4 PowerPlant demands {scenario} {year}')
    #plt.show()

    return allocated


if __name__ == '__main__':
    #pp_elec_to_nuts_h2(['T45-Strom']) # --> this was for dashboard data
    #exit(0)
    scenario = 'T45-H2'
    gas_type = 'H2'
    for year in [2030, 2045]:
        print(f'\nNodal allocation for {scenario} {year} {gas_type}')

        if gas_type == 'CH4':

            print('\nAllocating CH4 power plants...')

            allocated_tpp_ch4 = allocate_tpp_m(year, scenario)  # unit = MWh/h
            if not allocated_tpp_ch4.empty:
                print(f'\tTPP demands: {round(allocated_tpp_ch4.sum().sum() / 1_000_000, 1)} TWh')
                d = allocated_tpp_ch4.groupby(allocated_tpp_ch4.columns, axis=1).sum()
                with open(f'output/allocated/allocated_ch4_tpp_{scenario}_{year}.pkl', 'wb') as file:
                    pickle.dump(d, file)
            else:
                print('tpp is empty')

            print('\nAllocating CH4 industry...')
            rem_ch4, tso_ch4 = allocate_industry(scenario, year, 'CH4')
            if not tso_ch4.empty:
                d = tso_ch4.groupby(tso_ch4.columns, axis=1).sum()
                with open(f'output/allocated/allocated_ch4_ind_{scenario}_{year}.pkl', 'wb') as file:
                    pickle.dump(d, file)
                save_rem = rem_ch4.groupby(rem_ch4.columns, axis=1).sum()
                save_rem.to_excel(f'output/allocated/rem_CH4_dso_ind_{scenario}_{year}.xlsx')
            else:
                print('ind is empty')

            print('\nAllocating CH4 DSO demands...')
            dso_ch4 = allocate_dso(year, scenario, 'CH4', save_rem)
            if not dso_ch4.empty:
                d = dso_ch4.groupby(dso_ch4.columns, axis=1).sum()
                with open(f'output/allocated/allocated_ch4_dso_{scenario}_{year}.pkl', 'wb') as file:
                    pickle.dump(d, file)
            else:
                print('dso is empty')

        if gas_type == 'H2':
            print('\nAllocating H2 power plants and electrolysis...')
            allocated_pp, allocated_ee = allocate_pp_elec_h2(year, scenario)

            # just saving electrolyisis and power plants separately
            with open(f'output/allocated/allocated_h2_electrolysis_{scenario}_{year}.pkl', 'wb') as file:
                pickle.dump(allocated_ee, file)
            with open(f'output/allocated/allocated_h2_reelectrification_{scenario}_{year}.pkl', 'wb') as file:
                pickle.dump(allocated_pp, file)
            continue

            allocated_pp_ee = allocated_ee.add(allocated_pp, fill_value=0)
            if not allocated_pp_ee.empty:
                print(f'\tTPP demands: {round(allocated_pp_ee.sum().sum()/1_000_000, 1)} TWh')
                d = allocated_pp_ee.groupby(allocated_pp_ee.columns, axis=1).sum()
                with open(f'output/allocated/allocated_h2_tpp_{scenario}_{year}.pkl', 'wb') as file:
                    pickle.dump(d, file)
            else:
                print('No data for EE/PP')

            #allocated_tpp_h2 = allocate_tpp_electrolysis_h2(year, scenario)  # unit = MWh/h
            #if not allocated_tpp_h2.empty:
            #    print(f'\tTPP demands: {round(allocated_tpp_h2.sum().sum()/1_000_000, 1)} TWh')
            #    d = allocated_tpp_h2.groupby(allocated_tpp_h2.columns, axis=1).sum()
            #    with open(f'output/allocated/allocated_h2_tpp_{scenario}_{year}.pkl', 'wb') as file:
            #        pickle.dump(d, file)
            #else:
            #    print('tpp is empty')

            print('\nAllocating H2 industry...')
            rem_h2, tso_h2 = allocate_industry(scenario, year, 'H2')
            if not tso_h2.empty:
                d = tso_h2.groupby(tso_h2.columns, axis=1).sum()
                with open(f'output/allocated/allocated_h2_ind_{scenario}_{year}.pkl', 'wb') as file:
                    pickle.dump(d, file)
                save_rem = rem_h2.groupby(rem_h2.columns, axis=1).sum()
                save_rem.to_excel(f'output/allocated/rem_H2_dso_ind_{scenario}_{year}.xlsx')
            else:
                print('ind is empty')

            print('\nAllocating H2 DSO demands...')
            dso_h2 = allocate_dso(year, scenario, 'H2', rem_h2)
            if not dso_h2.empty:
                d = dso_h2.groupby(dso_h2.columns, axis=1).sum()
                with open(f'output/allocated/allocated_h2_dso_{scenario}_{year}.pkl', 'wb') as file:
                    pickle.dump(d, file)
            else:
                print('dso is empty')


