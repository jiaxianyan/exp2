import os
import copy
import dgl
import torch
import random
import scipy.sparse as sp
import numpy as np

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
    test_events = [int(line[0]) for line in lines_event[train_event_num:train_event_num + valid_event_num:]]
    return train_events, valid_events, test_events, num_events_total

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

def get_train_graph(g, e2m_dict, event_ids, num_members_total, num_events_total):
    src_pos,dst_pos,src_neg,dst_neg = [],[],[],[]
    for event_id in event_ids:
        members = e2m_dict[event_id]
        for member in members:
            src_pos.append(event_id)
            dst_pos.append(int(member))

    g.add_edges(src_pos,dst_pos)
    print(len(src_pos))
    print(g.num_edges())
    g.edata['e'] = torch.tensor([1] * len(src_pos))

    pos_edges = zip(src_pos,dst_pos)
    neg_num, src_neg, dst_neg = 0, [], []
    print('generate negative edges')
    while neg_num < len(src_pos):
        a = random.randint(num_members_total, num_members_total + num_events_total - 1)
        b = random.randint(0, num_members_total - 1)
        if (a,b) not in pos_edges:
            src_neg.append(a)
            dst_neg.append(b)
            neg_num += 1

    g_neg = dgl.DGLGraph((src_neg, dst_neg),num_nodes=g.number_of_nodes())
    return g,g_neg

def get_test_graph(g, e2m_dict, event_ids,num_members_total):
    src_pos, dst_pos = [], []
    adj = sp.coo_matrix((np.ones(len(src_pos)), (np.array(src_pos) - num_members_total, np.array(dst_pos))))
    adj_neg = 1 - adj.todense()
    src_neg, dst_neg = np.where(adj_neg != 0)
    for event_id in event_ids:
        members = e2m_dict[event_id]
        for member in members:
            src_pos.append(event_id)
            dst_pos.append(int(member))

    src = src_pos + src_neg
    dst = dst_pos + dst_neg
    g.add_edges(src,dst)
    g.edata['e'] = torch.tensor([1] * len(src_pos) + [0] * len(src_neg))
    return g

def get_meetup_biparticle_graph(config):
    train_events_ids, valid_events_ids, test_events_ids, num_events_total = get_event_train_valid_test_sets(config)
    num_members_total = get_member_info(config)
    e2m_dict = get_event2member_dict(config,num_members_total,num_events_total)

    m2e_dict_total = get_member2event_dict(num_members_total,e2m_dict,train_events_ids+valid_events_ids+test_events_ids)
    m2e_dict_train = get_member2event_dict(num_members_total,e2m_dict,train_events_ids)

    g = dgl.DGLGraph()
    g.add_nodes(num_events_total+num_members_total)
    # g = dgl.add_self_loop(g)
    labels = [0] * num_events_total + [1] * num_members_total
    g.ndata['h'] = torch.nn.functional.one_hot(torch.tensor(labels)).float()
    # get train graph

    g_train_pos,g_train_neg = get_train_graph(copy.deepcopy(g),e2m_dict,train_events_ids,num_members_total,num_events_total)
    # g_valid = get_test_graph(copy.deepcopy(g),e2m_dict,valid_events_ids,num_members_total)
    # g_test  = get_test_graph(copy.deepcopy(g),e2m_dict,test_events_ids,num_members_total)

    return g_train_pos, g_train_neg, valid_events_ids, test_events_ids