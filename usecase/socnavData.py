"""
This script generates the dataset.
"""

import os
import sys
import json
import copy
from collections import namedtuple
import math

import torch as th
import dgl
from dgl.data import DGLDataset
from dgl import save_graphs, load_graphs
from dgl.data.utils import save_info, load_info
import numpy as np

grid_width = 18  # 30 #18
output_width = 73  # 121 #73
area_width = 800.  # Spatial area of the grid

threshold_human_wall = 1.5
limit = 4000 #149999  # 31191

path_saves = 'saves/'
graphData = namedtuple('graphData', ['src_nodes', 'dst_nodes', 'n_nodes', 'features', 'edge_types',
                                     'edge_norms', 'position_by_id', 'typeMap', 'labels', 'w_segments'])

N_INTERVALS = 3
FRAMES_INTERVAL = 1.

MAX_ADV = 3.5
MAX_ROT = 4.
MAX_HUMANS = 15

#  human to wall distance
def dist_h_w(h, wall):
    hxpos = float(h['x']) / 100.
    hypos = float(h['y']) / 100.
    wxpos = float(wall.xpos) / 100.
    wypos = float(wall.ypos) / 100.
    return math.sqrt((hxpos - wxpos) * (hxpos - wxpos) + (hypos - wypos) * (hypos - wypos))

# Calculate the closet node in the grid to a given node by its coordinates
def closest_grid_node(grid_ids, w_a, w_i, x, y):
    c_x = int((x * (w_i / w_a) + (w_i / 2)))
    c_y = int((y * (w_i / w_a) + (w_i / 2)))
    if 0 <= c_x < grid_width and 0 <= c_y < grid_width:
        return grid_ids[c_x][c_y]
    return None


def closest_grid_nodes(grid_ids, w_a, w_i, r, x, y):
    c_x = int((x * (w_i / w_a) + (w_i / 2)))
    c_y = int((y * (w_i / w_a) + (w_i / 2)))
    cols, rows = (int(math.ceil(r * w_i / w_a)), int(math.ceil(r * w_i / w_a)))
    rangeC = list(range(-cols, cols + 1))
    rangeR = list(range(-rows, rows + 1))
    p_arr = [[c, r] for c in rangeC for r in rangeR]
    grid_nodes = []
    r_g = r * w_i / w_a
    for p in p_arr:
        if math.sqrt(p[0] * p[0] + p[1] * p[1]) <= r_g:
            if 0 <= (c_x + p[0]) < grid_width and 0 <= (c_y + p[1]) < grid_width:
                grid_nodes.append(grid_ids[c_x + p[0]][c_y + p[1]])

    return grid_nodes


def get_node_descriptor_header():
    # Node Descriptor Table
    node_descriptor_header = ['R', 'H', 'O', 'L', 'W',
                              'h_dist', 'h_dist2', 'h_ang_sin', 'h_ang_cos', 'h_orient_sin', 'h_orient_cos',
                              'o_dist', 'o_dist2', 'o_ang_sin', 'o_ang_cos', 'o_orient_sin', 'o_orient_cos',
                              'r_m_h', 'r_m_h2', 'r_hs', 'r_hs2',
                              'w_dist', 'w_dist2', 'w_ang_sin', 'w_ang_cos', 'w_orient_sin', 'w_orient_cos']
    return node_descriptor_header


def get_relations(alt):
    rels = None
    if alt == '1':
        rels = {'p_r', 'o_r', 'p_p', 'p_o', 'w_r', 'g_r', 'w_p'}
        # p = person
        # r = room
        # o = object
        # w = wall
        # g = goal
        for e in list(rels):
            rels.add(e[::-1])
        rels.add('self')
    elif alt == '2':  # With grid
        room_set = {'l_p', 'l_o', 'l_w', 'l_g', 'p_p', 'p_o', 'p_g', 'o_g', 'w_g', 'l_t', 'l_g', 'w_p', 'g_t'}
        grid_set = {'g_c', 'g_ri', 'g_le', 'g_u', 'g_d', 'g_uri', 'g_dri', 'g_ule', 'g_dle'}
        # ^
        # |_p = person             g_ri = grid right
        # |_w = wall               g_le = grid left
        # |_l = lounge             g_u = grid up
        # |_o = object             g_d = grid down
        # |_g = grid node
        # |_t = target (goal) node
        self_edges_set = {'P', 'O', 'W', 'L', 'T'}

        for e in list(room_set):
            room_set.add(e[::-1])
        rels = room_set | grid_set | self_edges_set

    rels = sorted(list(rels))
    num_rels = len(rels)

    return rels, num_rels


def get_features(alt):
    all_features = None
    time_one_hot = ['is_t_0', 'is_t_m1', 'is_t_m2']
    time_sequence_features = ['is_first_frame', 'time_left']
    human_metric_features = ['hum_x_pos', 'hum_y_pos', 'human_a_vel', 'human_x_vel', 'human_y_vel',
                             'hum_orientation_sin', 'hum_orientation_cos',
                             'hum_dist', 'hum_inv_dist']
    object_metric_features = ['obj_x_pos', 'obj_y_pos', 'obj_a_vel', 'obj_x_vel', 'obj_y_vel',
                              'obj_orientation_sin', 'obj_orientation_cos',
                              'obj_x_size', 'obj_y_size',
                              'obj_dist', 'obj_inv_dist']
    room_metric_features = ['room_humans', 'room_humans2']
    wall_metric_features = ['wall_x_pos', 'wall_y_pos', 'wall_orientation_sin', 'wall_orientation_cos',
                            'wall_dist', 'wall_inv_dist']
    goal_metric_features = ['goal_x_pos', 'goal_y_pos', 'goal_dist', 'goal_inv_dist']
    grid_metric_features = ['grid_x_pos', 'grid_y_pos']
    if alt == '1':
        node_types_one_hot = ['human', 'object', 'room', 'wall', 'goal']
        all_features = node_types_one_hot + time_one_hot + time_sequence_features + human_metric_features + \
                        object_metric_features + room_metric_features + wall_metric_features + goal_metric_features
    elif alt == '2':
        node_types_one_hot = ['human', 'object', 'room', 'wall', 'goal', 'grid']
        all_features = node_types_one_hot + time_one_hot + time_sequence_features + human_metric_features + \
                       object_metric_features + room_metric_features + wall_metric_features + goal_metric_features + \
                       grid_metric_features
    feature_dimensions = len(all_features)

    return all_features, feature_dimensions


#################################################################
# Different initialize alternatives:
#################################################################
# Function to generate the necessary data to create the grid graph
def generate_grid_graph_data():
    # Define variables for edge types and relations
    grid_rels, _ = get_relations('2')
    edge_types = []  # List to store the relation of each edge
    edge_norms = []  # List to store the norm of each edge

    # Grid properties
    connectivity = 8  # Connections of each node
    node_ids = np.zeros((grid_width, grid_width), dtype=int)  # Array to store the IDs of each node
    typeMap = dict()
    coordinates_gridGraph = dict()  # Dict to store the spatial coordinates of each node
    src_nodes = []  # List to store source nodes
    dst_nodes = []  # List to store destiny nodes

    # Feature dimensions
    all_features, n_features = get_features('2')

    # Compute the number of nodes and initialize feature vectors
    n_nodes = grid_width ** 2
    features_gridGraph = th.zeros(n_nodes, n_features)

    max_used_id = -1
    for y in range(grid_width):
        for x in range(grid_width):
            max_used_id += 1
            node_id = max_used_id
            node_ids[x][y] = node_id

            # Self edges
            src_nodes.append(node_id)
            dst_nodes.append(node_id)
            edge_types.append(grid_rels.index('g_c'))
            edge_norms.append([1.])

            if x < grid_width - 1:
                src_nodes.append(node_id)
                dst_nodes.append(node_id + 1)
                edge_types.append(grid_rels.index('g_ri'))
                edge_norms.append([1.])
                if connectivity == 8 and y > 0:
                    src_nodes.append(node_id)
                    dst_nodes.append(node_id - grid_width + 1)
                    edge_types.append(grid_rels.index('g_uri'))
                    edge_norms.append([1.])
            if x > 0:
                src_nodes.append(node_id)
                dst_nodes.append(node_id - 1)
                edge_types.append(grid_rels.index('g_le'))
                edge_norms.append([1.])
                if connectivity == 8 and y < grid_width - 1:
                    src_nodes.append(node_id)
                    dst_nodes.append(node_id + grid_width - 1)
                    edge_types.append(grid_rels.index('g_dle'))
                    edge_norms.append([1.])
            if y < grid_width - 1:
                src_nodes.append(node_id)
                dst_nodes.append(node_id + grid_width)
                edge_types.append(grid_rels.index('g_d'))
                edge_norms.append([1.])
                if connectivity == 8 and x < grid_width - 1:
                    src_nodes.append(node_id)
                    dst_nodes.append(node_id + grid_width + 1)
                    edge_types.append(grid_rels.index('g_dri'))
                    edge_norms.append([1.])
            if y > 0:
                src_nodes.append(node_id)
                dst_nodes.append(node_id - grid_width)
                edge_types.append(grid_rels.index('g_u'))
                edge_norms.append([1.])
                if connectivity == 8 and x > 0:
                    src_nodes.append(node_id)
                    dst_nodes.append(node_id - grid_width - 1)
                    edge_types.append(grid_rels.index('g_ule'))
                    edge_norms.append([1.])

            typeMap[node_id] = 'g'  # 'g' for 'grid'
            x_pos = (-area_width / 2. + (x + 0.5) * (area_width / grid_width))
            y_pos = (-area_width / 2. + (y + 0.5) * (area_width / grid_width))
            features_gridGraph[node_id, all_features.index('grid')] = 1
            features_gridGraph[node_id, all_features.index('grid_x_pos')] = 2 * x_pos / 1000
            features_gridGraph[node_id, all_features.index('grid_y_pos')] = 2 * y_pos / 1000

            coordinates_gridGraph[node_id] = [2 * x_pos / 1000, 2 * y_pos / 1000]

    src_nodes = th.LongTensor(src_nodes)
    dst_nodes = th.LongTensor(dst_nodes)

    edge_types = th.LongTensor(edge_types)
    edge_norms = th.Tensor(edge_norms)

    return src_nodes, dst_nodes, n_nodes, features_gridGraph, edge_types, edge_norms, coordinates_gridGraph, typeMap, \
           node_ids, None

# So far there is only one alternative implemented that I think is the most complete

def initializeAlt1(data, w_segments=[]):
    # Initialize variables
    rels, num_rels = get_relations('1')
    edge_types = []  # List to store the relation of each edge
    edge_norms = []  # List to store the norm of each edge
    max_used_id = 0  # Initialise id counter (0 for the robot)

    # Compute data for walls
    Wall = namedtuple('Wall', ['orientation', 'xpos', 'ypos'])
    walls = []
    i_w = 0
    for wall_segment in data['walls']:
        p1 = np.array([wall_segment["x1"], wall_segment["y1"]]) * 100
        p2 = np.array([wall_segment["x2"], wall_segment["y2"]]) * 100
        dist = np.linalg.norm(p1 - p2)
        if i_w >= len(w_segments):
            iters = int(dist / 400) + 1
            w_segments.append(iters)
        if w_segments[i_w] > 1:  # WE NEED TO CHECK THIS PART
            v = (p2 - p1) / w_segments[i_w]
            for i in range(w_segments[i_w]):
                pa = p1 + v * i
                pb = p1 + v * (i + 1)
                inc2 = pb - pa
                midsp = (pa + pb) / 2
                walls.append(Wall(math.atan2(inc2[0], inc2[1]), midsp[0], midsp[1]))
        else:
            inc = p2 - p1
            midp = (p2 + p1) / 2
            walls.append(Wall(math.atan2(inc[0], inc[1]), midp[0], midp[1]))
        i_w += 1

    # Compute the number of nodes
    # one for the robot + room walls   + humans    + objects              + room(global node)
    n_nodes = 1 + len(walls) + len(data['people']) + len(data['objects']) + 1

    # Feature dimensions
    all_features, n_features = get_features('1')
    features = th.zeros(n_nodes, n_features)

    # Nodes variables
    typeMap = dict()
    position_by_id = {}
    src_nodes = []  # List to store source nodes
    dst_nodes = []  # List to store destiny nodes

    # Labels
    labels = np.array([data['command'][0], data['command'][2]])
    labels[0] = labels[0] / MAX_ADV
    labels[1] = labels[1] / MAX_ROT

    # room (id 0)
    room_id = 0
    typeMap[room_id] = 'r'  # 'r' for 'room'
    position_by_id[room_id] = [0, 0]
    features[room_id, all_features.index('room')] = 1.
    features[room_id, all_features.index('room_humans')] = len(data['people']) / MAX_HUMANS
    features[room_id, all_features.index('room_humans2')] = (len(data['people']) ** 2) / (MAX_HUMANS ** 2)
    max_used_id += 1

    # humans
    for h in data['people']:
        src_nodes.append(h['id'])
        dst_nodes.append(room_id)
        edge_types.append(rels.index('p_r'))
        edge_norms.append([1. / len(data['people'])])

        src_nodes.append(room_id)
        dst_nodes.append(h['id'])
        edge_types.append(rels.index('r_p'))
        edge_norms.append([1.])

        typeMap[h['id']] = 'p'  # 'p' for 'person'
        max_used_id += 1
        xpos = h['x'] / 10.
        ypos = h['y'] / 10.
        position_by_id[h['id']] = [xpos, ypos]
        dist = math.sqrt(xpos ** 2 + ypos ** 2)
        va = h['va'] / 10.
        vx = h['vx'] / 10.
        vy = h['vy'] / 10.

        orientation = h['a']
        while orientation > math.pi:
            orientation -= 2. * math.pi
        while orientation < -math.pi:
            orientation += 2. * math.pi
        if orientation > math.pi:
            orientation -= math.pi
        elif orientation < -math.pi:
            orientation += math.pi

        # print(str(math.degrees(angle)) + ' ' + str(math.degrees(orientation)) + ' ' + str(math.degrees(angle_hum)))
        features[h['id'], all_features.index('human')] = 1.
        features[h['id'], all_features.index('hum_orientation_sin')] = math.sin(orientation)
        features[h['id'], all_features.index('hum_orientation_cos')] = math.cos(orientation)
        features[h['id'], all_features.index('hum_x_pos')] = xpos
        features[h['id'], all_features.index('hum_y_pos')] = ypos
        features[h['id'], all_features.index('human_a_vel')] = va
        features[h['id'], all_features.index('human_x_vel')] = vx
        features[h['id'], all_features.index('human_y_vel')] = vy
        features[h['id'], all_features.index('hum_dist')] = dist
        features[h['id'], all_features.index('hum_inv_dist')] = 1. - dist  # /(1.+dist*10.)


    # objects
    for o in data['objects']:
        src_nodes.append(o['id'])
        dst_nodes.append(room_id)
        edge_types.append(rels.index('o_r'))
        edge_norms.append([1.])

        src_nodes.append(room_id)
        dst_nodes.append(o['id'])
        edge_types.append(rels.index('r_o'))
        edge_norms.append([1.])

        typeMap[o['id']] = 'o'  # 'o' for 'object'
        max_used_id += 1
        xpos = o['x'] / 10.
        ypos = o['y'] / 10.
        position_by_id[o['id']] = [xpos, ypos]
        dist = math.sqrt(xpos ** 2 + ypos ** 2)
        va = o['va'] / 10.
        vx = o['vx'] / 10.
        vy = o['vy'] / 10.

        orientation = o['a']
        while orientation > math.pi:
            orientation -= 2. * math.pi
        while orientation < -math.pi:
            orientation += 2. * math.pi
        features[o['id'], all_features.index('object')] = 1
        features[o['id'], all_features.index('obj_orientation_sin')] = math.sin(orientation)
        features[o['id'], all_features.index('obj_orientation_cos')] = math.cos(orientation)
        features[o['id'], all_features.index('obj_x_pos')] = xpos
        features[o['id'], all_features.index('obj_y_pos')] = ypos
        features[o['id'], all_features.index('obj_a_vel')] = va
        features[o['id'], all_features.index('obj_x_vel')] = vx
        features[o['id'], all_features.index('obj_y_vel')] = vy
        features[o['id'], all_features.index('obj_x_size')] = o['size_x']
        features[o['id'], all_features.index('obj_y_size')] = o['size_y']
        features[o['id'], all_features.index('obj_dist')] = dist
        features[o['id'], all_features.index('obj_inv_dist')] = 1. - dist  # /(1.+dist*10.)

    # Goal
    goal_id = max_used_id
    typeMap[goal_id] = 'g'  # 'g' for 'goal'
    src_nodes.append(goal_id)
    dst_nodes.append(room_id)
    edge_types.append(rels.index('g_r'))
    edge_norms.append([1.])
    #edge_norms.append([1. / len(data['objects'])])

    src_nodes.append(room_id)
    dst_nodes.append(goal_id)
    edge_types.append(rels.index('r_g'))
    edge_norms.append([1.])

    xpos = data['goal'][0]['x'] / 10.
    ypos = data['goal'][0]['y'] / 10.
    position_by_id[goal_id] = [xpos, ypos]
    dist = math.sqrt(xpos ** 2 + ypos ** 2)
    features[goal_id, all_features.index('goal')] = 1
    features[goal_id, all_features.index('goal_x_pos')] = xpos
    features[goal_id, all_features.index('goal_y_pos')] = ypos
    features[goal_id, all_features.index('goal_dist')] = dist
    features[goal_id, all_features.index('goal_inv_dist')] = 1. - dist  # /(1.+dist*10.)

    max_used_id += 1

    # walls
    wids = dict()
    for wall in walls:
        wall_id = max_used_id
        wids[wall] = wall_id
        typeMap[wall_id] = 'w'  # 'w' for 'walls'
        max_used_id += 1

        src_nodes.append(wall_id)
        dst_nodes.append(room_id)
        edge_types.append(rels.index('w_r'))
        edge_norms.append([1. / len(walls)])

        src_nodes.append(room_id)
        dst_nodes.append(wall_id)
        edge_types.append(rels.index('r_w'))
        edge_norms.append([1.])

        position_by_id[wall_id] = [wall.xpos / 100., wall.ypos / 100.]
        dist = math.sqrt((wall.xpos / 1000.) ** 2 + (wall.ypos / 1000.) ** 2)

        features[wall_id, all_features.index('wall')] = 1.
        features[wall_id, all_features.index('wall_orientation_sin')] = math.sin(wall.orientation)
        features[wall_id, all_features.index('wall_orientation_cos')] = math.cos(wall.orientation)
        features[wall_id, all_features.index('wall_x_pos')] = wall.xpos / 1000.
        features[wall_id, all_features.index('wall_y_pos')] = wall.ypos / 1000.
        features[wall_id, all_features.index('wall_dist')] = dist
        features[wall_id, all_features.index('wall_inv_dist')] = 1. - dist  # 1./(1.+dist*10.)

    for h in data['people']:
        number = 0
        for wall in walls:
            dist = dist_h_w(h, wall)
            if dist < threshold_human_wall:
                number -= - 1
        for wall in walls:
            dist = dist_h_w(h, wall)
            if dist < threshold_human_wall:
                src_nodes.append(wids[wall])
                dst_nodes.append(h['id'])
                edge_types.append(rels.index('w_p'))
                edge_norms.append([1. / number])

    for wall in walls:
        number = 0
        for h in data['people']:
            dist = dist_h_w(h, wall)
            if dist < threshold_human_wall:
                number -= - 1
        for h in data['people']:
            dist = dist_h_w(h, wall)
            if dist < threshold_human_wall:
                src_nodes.append(h['id'])
                dst_nodes.append(wids[wall])
                edge_types.append(rels.index('p_w'))
                edge_norms.append([1. / number])

    # interaction links
    for link in data['interaction']:
        typeLdir = typeMap[link['src']] + '_' + typeMap[link['dst']]
        typeLinv = typeMap[link['dst']] + '_' + typeMap[link['src']]

        src_nodes.append(link['src'])
        dst_nodes.append(link['dst'])
        edge_types.append(rels.index(typeLdir))
        edge_norms.append([1.])

        src_nodes.append(link['dst'])
        dst_nodes.append(link['src'])
        edge_types.append(rels.index(typeLinv))
        edge_norms.append([1.])


    # self edges
    for node_id in range(n_nodes - 1):
        src_nodes.append(node_id)
        dst_nodes.append(node_id)
        edge_types.append(rels.index('self'))
        edge_norms.append([1.])

    # Convert outputs to tensors
    src_nodes = th.LongTensor(src_nodes)
    dst_nodes = th.LongTensor(dst_nodes)

    edge_types = th.LongTensor(edge_types)
    edge_norms = th.Tensor(edge_norms)

    return src_nodes, dst_nodes, n_nodes, features, edge_types, edge_norms, position_by_id, typeMap, labels, w_segments


def initializeAlt2(data, w_segments=[]):
    # Initialize variables
    rels, num_rels = get_relations('2')
    edge_types = []  # List to store the relation of each edge
    edge_norms = []  # List to store the norm of each edge
    max_used_id = 0  # Initialise id counter (0 for the robot)

    # Compute data for walls
    Wall = namedtuple('Wall', ['orientation', 'xpos', 'ypos'])
    walls = []
    i_w = 0
    for wall_segment in data['walls']:
        p1 = np.array([wall_segment["x1"], wall_segment["y1"]]) * 100
        p2 = np.array([wall_segment["x2"], wall_segment["y2"]]) * 100
        dist = np.linalg.norm(p1 - p2)
        if i_w >= len(w_segments):
            iters = int(dist / 400) + 1
            w_segments.append(iters)
        if w_segments[i_w] > 1:  # WE NEED TO CHECK THIS PART
            v = (p2 - p1) / w_segments[i_w]
            for i in range(w_segments[i_w]):
                pa = p1 + v * i
                pb = p1 + v * (i + 1)
                inc2 = pb - pa
                midsp = (pa + pb) / 2
                walls.append(Wall(math.atan2(inc2[0], inc2[1]), midsp[0], midsp[1]))
        else:
            inc = p2 - p1
            midp = (p2 + p1) / 2
            walls.append(Wall(math.atan2(inc[0], inc[1]), midp[0], midp[1]))
        i_w += 1

    # Compute the number of nodes
    # one for the robot + room walls   + humans    + objects              + room(global node)
    n_nodes = 1 + len(walls) + len(data['people']) + len(data['objects']) + 1

    # Feature dimensions
    all_features, n_features = get_features('2')
    features = th.zeros(n_nodes, n_features)

    # Nodes variables
    typeMap = dict()
    position_by_id = {}
    src_nodes = []  # List to store source nodes
    dst_nodes = []  # List to store destiny nodes

    # Labels
    labels = np.array(data['command'])
    labels[0] = labels[0] / MAX_ADV
    labels[2] = labels[2] / MAX_ROT

    # room (id 0)
    room_id = 0
    typeMap[room_id] = 'l'  # 'r' for 'lunge' (room) global node
    position_by_id[room_id] = [0, 0]
    features[room_id, all_features.index('room')] = 1.
    features[room_id, all_features.index('room_humans')] = len(data['people']) / MAX_HUMANS
    features[room_id, all_features.index('room_humans2')] = (len(data['people']) ** 2) / (MAX_HUMANS ** 2)
    max_used_id += 1

    # humans
    for h in data['people']:
        src_nodes.append(h['id'])
        dst_nodes.append(room_id)
        edge_types.append(rels.index('p_l'))
        edge_norms.append([1. / len(data['people'])])

        src_nodes.append(room_id)
        dst_nodes.append(h['id'])
        edge_types.append(rels.index('l_p'))
        edge_norms.append([1.])

        typeMap[h['id']] = 'p'  # 'p' for 'person'
        max_used_id += 1
        xpos = h['x'] / 10.
        ypos = h['y'] / 10.
        position_by_id[h['id']] = [xpos*10, ypos*10]
        dist = math.sqrt(xpos ** 2 + ypos ** 2)
        va = h['va'] / 10.
        vx = h['vx'] / 10.
        vy = h['vy'] / 10.

        orientation = h['a']
        while orientation > math.pi:
            orientation -= 2. * math.pi
        while orientation < -math.pi:
            orientation += 2. * math.pi
        if orientation > math.pi:
            orientation -= math.pi
        elif orientation < -math.pi:
            orientation += math.pi

        # print(str(math.degrees(angle)) + ' ' + str(math.degrees(orientation)) + ' ' + str(math.degrees(angle_hum)))
        features[h['id'], all_features.index('human')] = 1.
        features[h['id'], all_features.index('hum_orientation_sin')] = math.sin(orientation)
        features[h['id'], all_features.index('hum_orientation_cos')] = math.cos(orientation)
        features[h['id'], all_features.index('hum_x_pos')] = xpos
        features[h['id'], all_features.index('hum_y_pos')] = ypos
        features[h['id'], all_features.index('human_a_vel')] = va
        features[h['id'], all_features.index('human_x_vel')] = vx
        features[h['id'], all_features.index('human_y_vel')] = vy
        features[h['id'], all_features.index('hum_dist')] = dist
        features[h['id'], all_features.index('hum_inv_dist')] = 1. - dist  # /(1.+dist*10.)


    # objects
    for o in data['objects']:
        src_nodes.append(o['id'])
        dst_nodes.append(room_id)
        edge_types.append(rels.index('o_l'))
        edge_norms.append([1.])

        src_nodes.append(room_id)
        dst_nodes.append(o['id'])
        edge_types.append(rels.index('l_o'))
        edge_norms.append([1.])

        typeMap[o['id']] = 'o'  # 'o' for 'object'
        max_used_id += 1
        xpos = o['x'] / 10.
        ypos = o['y'] / 10.
        position_by_id[o['id']] = [xpos*10, ypos*10]
        dist = math.sqrt(xpos ** 2 + ypos ** 2)
        va = o['va'] / 10.
        vx = o['vx'] / 10.
        vy = o['vy'] / 10.

        orientation = o['a']
        while orientation > math.pi:
            orientation -= 2. * math.pi
        while orientation < -math.pi:
            orientation += 2. * math.pi
        features[o['id'], all_features.index('object')] = 1
        features[o['id'], all_features.index('obj_orientation_sin')] = math.sin(orientation)
        features[o['id'], all_features.index('obj_orientation_cos')] = math.cos(orientation)
        features[o['id'], all_features.index('obj_x_pos')] = xpos
        features[o['id'], all_features.index('obj_y_pos')] = ypos
        features[o['id'], all_features.index('obj_a_vel')] = va
        features[o['id'], all_features.index('obj_x_vel')] = vx
        features[o['id'], all_features.index('obj_y_vel')] = vy
        features[o['id'], all_features.index('obj_x_size')] = o['size_x']
        features[o['id'], all_features.index('obj_y_size')] = o['size_y']
        features[o['id'], all_features.index('obj_dist')] = dist
        features[o['id'], all_features.index('obj_inv_dist')] = 1. - dist  # /(1.+dist*10.)

    # Goal
    goal_id = max_used_id
    typeMap[goal_id] = 't'  # 't' for 'target' (goal)
    src_nodes.append(goal_id)
    dst_nodes.append(room_id)
    edge_types.append(rels.index('t_l'))
    edge_norms.append([1.])
    #edge_norms.append([1. / len(data['objects'])])

    src_nodes.append(room_id)
    dst_nodes.append(goal_id)
    edge_types.append(rels.index('l_t'))
    edge_norms.append([1.])

    xpos = data['goal'][0]['x'] / 10.
    ypos = data['goal'][0]['y'] / 10.
    position_by_id[goal_id] = [xpos*10, ypos*10]
    dist = math.sqrt(xpos ** 2 + ypos ** 2)
    features[goal_id, all_features.index('goal')] = 1
    features[goal_id, all_features.index('goal_x_pos')] = xpos
    features[goal_id, all_features.index('goal_y_pos')] = ypos
    features[goal_id, all_features.index('goal_dist')] = dist
    features[goal_id, all_features.index('goal_inv_dist')] = 1. - dist  # /(1.+dist*10.)

    max_used_id += 1

    # walls
    wids = dict()
    for wall in walls:
        wall_id = max_used_id
        wids[wall] = wall_id
        typeMap[wall_id] = 'w'  # 'w' for 'walls'
        max_used_id += 1

        src_nodes.append(wall_id)
        dst_nodes.append(room_id)
        edge_types.append(rels.index('w_l'))
        edge_norms.append([1. / len(walls)])

        src_nodes.append(room_id)
        dst_nodes.append(wall_id)
        edge_types.append(rels.index('l_w'))
        edge_norms.append([1.])

        position_by_id[wall_id] = [wall.xpos / 100., wall.ypos / 100.]
        dist = math.sqrt((wall.xpos / 1000.) ** 2 + (wall.ypos / 1000.) ** 2)

        features[wall_id, all_features.index('wall')] = 1.
        features[wall_id, all_features.index('wall_orientation_sin')] = math.sin(wall.orientation)
        features[wall_id, all_features.index('wall_orientation_cos')] = math.cos(wall.orientation)
        features[wall_id, all_features.index('wall_x_pos')] = wall.xpos / 1000.
        features[wall_id, all_features.index('wall_y_pos')] = wall.ypos / 1000.
        features[wall_id, all_features.index('wall_dist')] = dist
        features[wall_id, all_features.index('wall_inv_dist')] = 1. - dist  # 1./(1.+dist*10.)

    for h in data['people']:
        number = 0
        for wall in walls:
            dist = dist_h_w(h, wall)
            if dist < threshold_human_wall:
                number -= - 1
        for wall in walls:
            dist = dist_h_w(h, wall)
            if dist < threshold_human_wall:
                src_nodes.append(wids[wall])
                dst_nodes.append(h['id'])
                edge_types.append(rels.index('w_p'))
                edge_norms.append([1. / number])

    for wall in walls:
        number = 0
        for h in data['people']:
            dist = dist_h_w(h, wall)
            if dist < threshold_human_wall:
                number -= - 1
        for h in data['people']:
            dist = dist_h_w(h, wall)
            if dist < threshold_human_wall:
                src_nodes.append(h['id'])
                dst_nodes.append(wids[wall])
                edge_types.append(rels.index('p_w'))
                edge_norms.append([1. / number])

    # interaction links
    for link in data['interaction']:
        typeLdir = typeMap[link['src']] + '_' + typeMap[link['dst']]
        typeLinv = typeMap[link['dst']] + '_' + typeMap[link['src']]

        src_nodes.append(link['src'])
        dst_nodes.append(link['dst'])
        edge_types.append(rels.index(typeLdir))
        edge_norms.append([1.])

        src_nodes.append(link['dst'])
        dst_nodes.append(link['src'])
        edge_types.append(rels.index(typeLinv))
        edge_norms.append([1.])

    # self edges
    for node_id in range(n_nodes):
        r_type = typeMap[node_id].upper()

        src_nodes.append(node_id)
        dst_nodes.append(node_id)
        edge_types.append(rels.index(r_type))
        edge_norms.append([1.])

    # Convert outputs to tensors
    src_nodes = th.LongTensor(src_nodes)
    dst_nodes = th.LongTensor(dst_nodes)

    edge_types = th.LongTensor(edge_types)
    edge_norms = th.Tensor(edge_norms)

    return src_nodes, dst_nodes, n_nodes, features, edge_types, edge_norms, position_by_id, typeMap, labels, w_segments


#################################################################
# Class to load the dataset
#################################################################

class GenerateDataset(DGLDataset):
    def __init__(self, path, alt, mode='train', raw_dir='data/', init_line=-1, end_line=-1, loc_limit=limit,
                 force_reload=False, verbose=True, debug=False, i_frame=0, assign_data=False):
        if type(path) is str:
            self.path = raw_dir + path
        else:
            self.path = path
        self.mode = mode
        self.alt = alt
        self.init_line = init_line
        self.end_line = end_line
        self.graphs = []
        self.labels = []
        self.data = dict()
        self.grid_data = None
        self.data['typemaps'] = []
        self.data['coordinates'] = []
        self.data['n_frames'] = []
        self.debug = debug
        self.limit = loc_limit
        self.assign_data = assign_data
        self.i_frame = i_frame
        self.force_reload = force_reload

        if self.mode == 'test':
            self.force_reload = True

        if self.alt == '1':
            self.dataloader = initializeAlt1
        elif self.alt == '2':
            self.dataloader = initializeAlt2
            self.grid_data = graphData(*generate_grid_graph_data())
        else:
            print('Introduce a valid initialize alternative')
            sys.exit(-1)

        # Define device.
        self.device = 'cpu'

        if self.debug:
            self.limit = 1 + (0 if init_line == -1 else init_line)

        super(GenerateDataset, self).__init__(self.mode, raw_dir=raw_dir, force_reload=self.force_reload, verbose=verbose)

    def get_dataset_name(self):
        graphs_path = 'graphs_' + self.mode + '_alt_' + self.alt + '_s_' + str(limit) + '.bin'
        info_path = 'info_' + self.mode + '_alt_' + self.alt + '_s_' + str(limit) + '.pkl'
        return graphs_path, info_path

    def merge_graphs(self, graphs_in_interval):
        all_features, n_features = get_features(self.alt)
        new_features = ['is_t_0', 'is_t_m1', 'is_t_m2']
        f_list = []
        src_list = []
        dst_list = []
        edge_types_list = []
        edge_norms_list = []
        typeMap = dict()
        coordinates = dict()
        n_nodes = 0
        rels, num_rels = get_relations(self.alt)
        g_i = 0
        offset = graphs_in_interval[0].n_nodes
        for g in graphs_in_interval:
            # Alternatives with grid
            if self.grid_data is not None:
                # Shift IDs of the typemap and coordinates lists
                for key in g.typeMap:
                    typeMap[key + len(self.grid_data.typeMap) + (offset * g_i)] = g.typeMap[key]
                    coordinates[key + len(self.grid_data.position_by_id) + (offset * g_i)] = g.position_by_id[key]

                if g_i == 0:
                    # Add links and their labels of grid graph and first room graph
                    src_list.append(self.grid_data.src_nodes)
                    src_list.append(g.src_nodes + self.grid_data.n_nodes)
                    dst_list.append(self.grid_data.dst_nodes)
                    dst_list.append(g.dst_nodes + self.grid_data.n_nodes)

                    edge_types_list.append(self.grid_data.edge_types)
                    edge_types_list.append(g.edge_types)
                    edge_norms_list.append(self.grid_data.edge_norms)
                    edge_norms_list.append(g.edge_norms)

                    # Add features of the nodes of the mentioned graphs
                    f_list.append(self.grid_data.features)
                    f_list.append(g.features)

                    # Add grid graph coordinates and typemaps
                    coordinates = {**self.grid_data.position_by_id, **coordinates}
                    typeMap = {**self.grid_data.typeMap, **typeMap}

                    # Update number of nodes
                    n_nodes = g.n_nodes + self.grid_data.n_nodes

                elif g_i > 0:
                    # Add edges and features of the new graph.
                    f_list.append(g.features)
                    src_list.append(g.src_nodes + (offset * g_i) + self.grid_data.n_nodes)
                    dst_list.append(g.dst_nodes + (offset * g_i) + self.grid_data.n_nodes)
                    edge_types_list.append(g.edge_types)
                    edge_norms_list.append(g.edge_norms)

                    # Temporal connections and edges labels
                    new_src_list = []
                    new_dst_list = []
                    new_etypes_list = []
                    new_enorms_list = []
                    for node in range(g.n_nodes):
                        new_src_list.append(node + (offset * (g_i - 1)) + self.grid_data.n_nodes)
                        new_dst_list.append(node + (offset * g_i) + self.grid_data.n_nodes)
                        new_etypes_list.append(num_rels + (g_i - 1) * 2)
                        new_enorms_list.append([1.])

                        new_src_list.append(node + (offset * g_i) + self.grid_data.n_nodes)
                        new_dst_list.append(node + (offset * (g_i - 1)) + self.grid_data.n_nodes)
                        new_etypes_list.append(num_rels + (g_i - 1) * 2 + 1)
                        new_enorms_list.append([1.])

                    src_list.append(th.IntTensor(new_src_list))
                    dst_list.append(th.IntTensor(new_dst_list))
                    edge_types_list.append(th.LongTensor(new_etypes_list))
                    edge_norms_list.append(th.Tensor(new_enorms_list))

                    #  Update number of nodes
                    n_nodes += g.n_nodes

                # Add new features for the graphs of the frames
                for f in new_features:
                    if g_i == new_features.index(f):
                        g.features[:, all_features.index(f)] = 1
                    else:
                        g.features[:, all_features.index(f)] = 0

                # Connect each graph frame to the grid:
                for r_n_id in range(1, g.n_nodes):
                    r_n_type = g.typeMap[r_n_id]
                    x, y = g.position_by_id[r_n_id]
                    closest_grid_nodes_id = closest_grid_nodes(self.grid_data.labels, area_width*2, grid_width,
                                                               25., x*100, y*100)
                    for g_id in closest_grid_nodes_id:
                        src_list.append(th.IntTensor([g_id]))
                        dst_list.append(th.IntTensor([r_n_id + self.grid_data.n_nodes + (offset * g_i)]))
                        edge_types_list.append(th.LongTensor([rels.index('g_' + r_n_type)]))
                        edge_norms_list.append(th.Tensor([[1.]]))

                        src_list.append(th.IntTensor([r_n_id + self.grid_data.n_nodes + (offset * g_i)]))
                        dst_list.append(th.IntTensor([g_id]))
                        edge_types_list.append(th.LongTensor([rels.index(r_n_type + '_g')]))
                        edge_norms_list.append(th.Tensor([[1.]]))

            # Alternatives without grid
            else:
                # Shift IDs of the typemap and coordinates lists
                for key in g.typeMap:
                    typeMap[key + (offset * g_i)] = g.typeMap[key]
                    coordinates[key + (offset * g_i)] = g.position_by_id[key]
                n_nodes += g.n_nodes
                f_list.append(g.features)
                # Add temporal edges
                src_list.append(g.src_nodes + (offset * g_i))
                dst_list.append(g.dst_nodes + (offset * g_i))
                edge_types_list.append(g.edge_types)
                edge_norms_list.append(g.edge_norms)
                if g_i > 0:
                    # Temporal connections and edges labels
                    new_src_list = []
                    new_dst_list = []
                    new_etypes_list = []
                    new_enorms_list = []
                    for node in range(g.n_nodes):
                        new_src_list.append(node + offset * (g_i - 1))
                        new_dst_list.append(node + offset * g_i)
                        new_etypes_list.append(num_rels + (g_i - 1) * 2)
                        new_enorms_list.append([1.])
                        new_src_list.append(node + offset * g_i)
                        new_dst_list.append(node + offset * (g_i - 1))
                        new_etypes_list.append(num_rels + (g_i - 1) * 2 + 1)
                        new_enorms_list.append([1.])
                    src_list.append(th.IntTensor(new_src_list))
                    dst_list.append(th.IntTensor(new_dst_list))
                    edge_types_list.append(th.LongTensor(new_etypes_list))
                    edge_norms_list.append(th.Tensor(new_enorms_list))
                for f in new_features:
                    if g_i == new_features.index(f):
                        g.features[:, all_features.index(f)] = 1
                    else:
                        g.features[:, all_features.index(f)] = 0
            g_i += 1

        return src_list, dst_list, edge_types_list, edge_norms_list, n_nodes, f_list, typeMap, coordinates

    def final_graph_from_json(self, ds_file, linen):
        all_features, n_features = get_features(self.alt)
        frames_in_interval = []
        graphs_in_interval = []

        print(ds_file)
        with open(ds_file) as json_file:
            data = json.load(json_file)

        frame_new = data[0]
        frames_in_interval.append(frame_new)
        graph_new_data = graphData(*self.dataloader(frame_new))
        w_segments = graph_new_data.w_segments
        graph_new_data.features[:, all_features.index('is_first_frame')] = 1
        graph_new_data.features[:, all_features.index('time_left')] = 1
        graphs_in_interval.append(graph_new_data)
        i_frame = 0
        for frame in data[1:]:
            frame_new = frame
            i_frame += 1
            if frame_new['timestamp'] - frames_in_interval[0]['timestamp'] < FRAMES_INTERVAL:  # Truncated to N seconds
                continue
            if linen % 1000 == 0:
                print(linen)
            if linen + 1 >= limit:
                print('Stop including more samples to speed up dataset loading')
                break
            linen += 1
            if self.init_line >= 0 and linen < self.init_line:
                continue
            if linen > self.end_line >= 0:
                continue

            graph_new_data = graphData(*self.dataloader(frame_new, w_segments))
            graph_new_data.features[:, all_features.index('time_left')] = 1. / (i_frame + 1.)
            frames_in_interval.insert(0, frame_new)
            graphs_in_interval.insert(0, graph_new_data)
            if len(graphs_in_interval) > N_INTERVALS:
                graphs_in_interval.pop(N_INTERVALS)
                frames_in_interval.pop(N_INTERVALS)

            src_list, dst_list, edge_types_list, edge_norms_list, n_nodes, f_list, typeMap, coordinates = \
                self.merge_graphs(graphs_in_interval)

            try:
                # Create merged graph:
                src_nodes = th.cat(src_list, dim=0)
                dst_nodes = th.cat(dst_list, dim=0)
                edge_types = th.cat(edge_types_list, dim=0)
                edge_norms = th.cat(edge_norms_list, dim=0)
                final_graph = dgl.graph((src_nodes, dst_nodes),
                                        num_nodes=n_nodes,
                                        idtype=th.int32, device=self.device)

                # Add merged features and update edge labels:
                final_graph.ndata['h'] = th.cat(f_list, dim=0).to(self.device)
                final_graph.edata.update({'rel_type': edge_types, 'norm': edge_norms})

                # Append final data
                self.graphs.append(final_graph)
                self.labels.append(graphs_in_interval[0].labels)
                self.data['typemaps'].append(typeMap)
                self.data['coordinates'].append(coordinates)
                self.data['n_frames'].append(len(graphs_in_interval))

            except Exception:
                print(frame)
                raise
        return linen

    def load_one_graph(self, path, i_frame):
        graph_data = graphData(*self.dataloader(path[0]))
        w_segments = graph_data.w_segments
        
        graphs_in_interval = [graph_data]
        for i in range(1, len(path)):
            graphs_in_interval.append(graphData(*self.dataloader(path[i], w_segments)))

        src_list, dst_list, edge_types_list, edge_norms_list, n_nodes, f_list, typeMap, coordinates = \
            self.merge_graphs(graphs_in_interval)

        try:
            # Create merged graph:
            src_nodes = th.cat(src_list, dim=0)
            dst_nodes = th.cat(dst_list, dim=0)
            edge_types = th.cat(edge_types_list, dim=0)
            edge_norms = th.cat(edge_norms_list, dim=0)
            final_graph = dgl.graph((src_nodes, dst_nodes),
                                    num_nodes=n_nodes,
                                    idtype=th.int32, device=self.device)

            # Add merged features and update edge labels:
            final_graph.ndata['h'] = th.cat(f_list, dim=0).to(self.device)
            final_graph.edata.update({'rel_type': edge_types, 'norm': edge_norms})

            # Append final data
            self.graphs.append(final_graph)
            self.labels.append(graphs_in_interval[0].labels)
            self.data['typemaps'].append(typeMap)
            self.data['coordinates'].append(coordinates)
            self.data['n_frames'].append(len(graphs_in_interval))
        except Exception:
            print("Error loading one graph")
            raise

    #################################################################
    # Implementation of abstract methods
    #################################################################

    def download(self):
        # No need to download any data
        pass

    def process(self):
        if type(self.path) is str and self.path.endswith('.json'):
            linen = -1
            if self.debug:
                self.final_graph_from_json(self.path, linen)
            else:
                with open(self.path) as json_file:
                    data = json.load(json_file)
                for frame in data:
                    if linen % 1000 == 0:
                        print(linen)
                    if linen + 1 >= limit:
                        print('Stop including more samples to speed up dataset loading')
                        break
                    linen += 1
                    if self.init_line >= 0 and linen < self.init_line:
                        continue
                    if linen > self.end_line >= 0:
                        continue
                    try:
                        graph_data = graphData(*self.dataloader(frame))
                        graph = dgl.graph((graph_data.src_nodes, graph_data.dst_nodes), num_nodes=graph_data.n_nodes,
                                          idtype=th.int32, device=self.device)
                        self.graphs.append(graph)
                        self.labels.append(graph_data.labels)
                        self.data['typemaps'].append(graph_data.typeMap)
                        self.data['coordinates'].append(graph_data.position_by_id)
                    except Exception:
                        print(frame)
                        raise
        elif type(self.path) is str and self.path.endswith('.txt'):
            linen = -1
            print(self.path)
            with open(self.path) as set_file:
                ds_files = set_file.read().splitlines()
            print("number of files for ", self.path, len(ds_files))

            for ds in ds_files:
                linen = self.final_graph_from_json(ds, linen)
                if linen + 1 >= limit:
                    break
        elif self.assign_data:
            for g in self.path:
                self.graphs.append(g.graphs[0])
                self.labels.append(g.labels[0])
                self.data.append(g.data[0])

        # elif type(self.path) == list and type(self.path[0]) == str:
        #     graph_data = graphData(*self.dataloader(json.loads(self.path[0])))
        #     graph = dgl.graph((graph_data.src_nodes, graph_data.dst_nodes), num_nodes=graph_data.n_nodes,
        #                       idtype=th.int32, device=self.device)
        #     self.graphs.append(graph)
        #     self.labels.append(graph_data.labels)
        #     self.data['typemaps'].append(graph_data.typeMap)
        #     self.data['coordinates'].append(graph_data.position_by_id)

        elif type(self.path) == list and len(self.path) >= 1:
            self.load_one_graph(self.path, self.i_frame)
            # self.data.append(RoomGraph(self.path, alt, mode))

        self.labels = th.tensor(self.labels, dtype=th.float64)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        return len(self.graphs)

    def save(self):
        if self.debug:
            return
        # Generate paths
        graphs_path, info_path = tuple((path_saves + x) for x in self.get_dataset_name())
        os.makedirs(os.path.dirname(path_saves), exist_ok=True)

        # Save graphs
        save_graphs(graphs_path, self.graphs, {'labels': self.labels})

        # Save additional info
        save_info(info_path, {'typemaps': self.data['typemaps'],
                              'coordinates': self.data['coordinates']})

    def load(self):
        # Generate paths
        graphs_path, info_path = tuple((path_saves + x) for x in self.get_dataset_name())

        # Load graphs
        self.graphs, label_dict = load_graphs(graphs_path)
        self.labels = label_dict['labels']

        # Load info
        self.data['typemaps'] = load_info(info_path)['typemaps']
        self.data['coordinates'] = load_info(info_path)['coordinates']

    def has_cache(self):
        # Generate paths
        graphs_path, info_path = tuple((path_saves + x) for x in self.get_dataset_name())
        if self.debug:
            return False
        return os.path.exists(graphs_path) and os.path.exists(info_path)
