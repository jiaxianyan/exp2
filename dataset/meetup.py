import os
import copy
import dgl
import torch
import pickle
import numpy as np
from tqdm import tqdm

def get_event_train_valid_test_sets(config):
    with open(config.data.event_id_path,'r') as f:
        lines_event = f.read().strip().split('\n')
    num_events_total = len(lines_event)
    lines_event = [line.split() for line in lines_event]
    lines_event.sort(key=lambda x:x[2])
    train_event_num = int(num_events_total * config.data.train_ratio)
    valid_event_num = int(num_events_total * config.data.valid_ratio)

    train_events = [int(line[0]) for line in lines_event[:train_event_num]]
    valid_events = [int(line[0]) for line in lines_event[train_event_num:train_event_num + valid_event_num]]
    test_events = [int(line[0]) for line in lines_event[train_event_num + valid_event_num:]]
    return train_events, valid_events, test_events, num_events_total

def get_time_event_sets(config):
    with open(config.data.event_id_path,'r') as f:
        lines_event = f.read().strip().split('\n')
    num_events_total = len(lines_event)
    lines_event = [line.split() for line in lines_event]
    lines_event.sort(key=lambda x:x[2])

    ratio_index = [int( ( i/config.data.time_sequence_length ) * num_events_total ) for i in range(config.data.time_sequence_length)]
    ratio_index.append(num_events_total)

    time_event_ids_list = []
    for i in range(config.data.time_sequence_length):
        time_event_ids_list.append([int(line[0]) for line in lines_event[ratio_index[i]:ratio_index[i+1]]])

    return time_event_ids_list,num_events_total

def get_member_info(config):
    with open(config.data.member_id_path,'r') as f:
        lines_member = f.read().strip().split('\n')
    num_members_total = len(lines_member)
    return num_members_total

def get_event2member_dict(config,num_members_total,num_events_total):
    e2m_dict = {}
    for i in range(num_events_total):
        e2m_dict[i+num_members_total] = []
    with open(config.data.event2member_path,'r') as f:
        lines = f.read().strip().split('\n')
    lines = [line.split() for line in lines]
    for line in lines:
        e2m_dict[int(line[0])] = line[1:]
    return e2m_dict

def get_member2event_dict(num_members_total,e2m_dict,event_ids):
    m2e_dict = {}
    for i in range(num_members_total):
        m2e_dict[i] = []
    for event_id in event_ids:
        members = e2m_dict[event_id]
        for member in members:
            m2e_dict[int(member)].append(event_id)
    return m2e_dict

def get_event2gruop_dict(config):
    e2g_dict = {}
    with open(config.data.event2group_path,'r') as f:
        lines = f.read().strip().split('\n')
    lines = [line.split() for line in lines]
    for line in lines:
        e2g_dict[int(line[0])] = int(line[1])
    return e2g_dict

def get_group2member_dict(config):
    g2m_dict = {}
    with open(config.data.group2member_path,'r') as f:
        lines = f.read().strip().split('\n')
    lines = [line.split() for line in lines]
    for line in lines:
        g2m_dict[int(line[0])] = line[1:]
    return g2m_dict

def get_train_graph(g, e2m_dict, e2g_dict, g2m_dict, event_ids):
    src_pos,dst_pos,src_neg,dst_neg = [],[],[],[]
    g_pos, g_neg = copy.deepcopy(g), copy.deepcopy(g)
    for event_id in tqdm(event_ids):
        gruop_id = e2g_dict[event_id]
        if gruop_id not in g2m_dict.keys():
            continue
        group_members = g2m_dict[gruop_id]
        attend_members = e2m_dict[event_id]
        absent_members = list(set(group_members) - set(attend_members))

        if len(absent_members) > len(attend_members):
            neg_ids = np.random.choice(len(absent_members), len(attend_members) + 1)
        else:
            neg_ids = range(len(absent_members))
        src_pos.extend([event_id] * len(attend_members))
        dst_pos.extend([int(member) for member in attend_members])

        src_neg.extend([event_id] * len(neg_ids))
        dst_neg.extend([int(absent_members[neg_id]) for neg_id in neg_ids])

    g_pos.add_edges(src_pos, dst_pos)
    g_neg.add_edges(src_neg, dst_neg)
    return g_pos,g_neg

def get_meetup_biparticle_graph(config):
    train_events_ids, valid_events_ids, test_events_ids, num_events_total = get_event_train_valid_test_sets(config)
    num_members_total = get_member_info(config)
    e2m_dict = get_event2member_dict(config,num_members_total,num_events_total)
    e2g_dict = get_event2gruop_dict(config)
    g2m_dict = get_group2member_dict(config)

    g = dgl.DGLGraph()
    g.add_nodes(num_events_total+num_members_total)
    labels = [0] * num_events_total + [1] * num_members_total
    g.ndata['h'] = torch.nn.functional.one_hot(torch.tensor(labels)).float()

    g_train_pos, g_train_neg = get_train_graph(g, e2m_dict, e2g_dict, g2m_dict, train_events_ids)
    g_valid_pos, g_valid_neg = get_train_graph(g, e2m_dict, e2g_dict, g2m_dict, valid_events_ids)
    g_test_pos, g_test_neg = get_train_graph(g, e2m_dict, e2g_dict, g2m_dict, test_events_ids)
    return (g_train_pos, g_train_neg, g_valid_pos, g_valid_neg, g_test_pos, g_test_neg),\
           e2m_dict, e2g_dict, g2m_dict, num_members_total

def get_meetup_biparticle_graph_sequence(config):
    time_event_ids_list,num_events_total = get_time_event_sets(config)
    num_members_total = get_member_info(config)
    e2m_dict = get_event2member_dict(config,num_members_total,num_events_total)
    e2g_dict = get_event2gruop_dict(config)
    g2m_dict = get_group2member_dict(config)

    if os.path.exists('graphs.pkl'):
        with open('graphs.pkl','rb') as f:
            graph_sets = pickle.load(f)
    else:
        g = dgl.DGLGraph()
        g.add_nodes(num_events_total+num_members_total)
        labels = [0] * num_events_total + [1] * num_members_total
        g.ndata['h'] = torch.nn.functional.one_hot(torch.tensor(labels)).float()

        train_events_ids, valid_events_ids, test_events_ids = [], [], []
        graph_sets = []
        for index,event_ids in enumerate(time_event_ids_list[:-3]):
            train_events_ids += event_ids
            valid_events_ids = time_event_ids_list[index+1]
            test_events_ids = time_event_ids_list[index+2] + time_event_ids_list[index+3]
            g_train_pos, g_train_neg = get_train_graph(g, e2m_dict, e2g_dict, g2m_dict, train_events_ids)
            g_valid_pos, g_valid_neg = get_train_graph(g, e2m_dict, e2g_dict, g2m_dict, valid_events_ids)
            g_test_pos, g_test_neg = get_train_graph(g, e2m_dict, e2g_dict, g2m_dict, test_events_ids)
            graph_sets.append((g_train_pos, g_train_neg, g_valid_pos, g_valid_neg, g_test_pos, g_test_neg))
        with open('graphs.pkl','wb') as f:
            pickle.dump(graph_sets,f)

    return graph_sets, e2m_dict, e2g_dict, g2m_dict, num_members_total
