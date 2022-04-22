import os
import dgl
import torch

def get_event_train_valid_test_sets(config):
    with open(config.data.event_id_path,'r') as f:
        lines_event = f.read().strip().split('\n')
    num_events_total = len(lines_event)
    lines_event = [line.split() for line in lines_event]
    lines_event.sort(key=lambda x:x[2])
    train_event_num = int(num_events_total * config.data.train_ratio)
    valid_event_num = int(num_events_total * config.data.valid_ratio)

    train_events = [int(line[0]) - 1 for line in lines_event[:train_event_num]]
    valid_events = [int(line[0]) - 1 for line in lines_event[train_event_num:train_event_num + valid_event_num]]
    test_events = [int(line[0]) - 1 for line in lines_event[train_event_num:train_event_num + valid_event_num:]]
    return train_events, valid_events, test_events, num_events_total

def get_member_info(config):
    with open(config.data.member_id_path,'r') as f:
        lines_member = f.read().strip().split('\n')
    num_members_total = len(lines_member)
    return num_members_total

def get_event2member_dict(config,num_members_total,num_events_total):
    e2m_dict = {}
    for i in range(num_events_total):
        e2m_dict[str(i+num_members_total)] = []
    with open(config.data.event2member_path,'r') as f:
        lines = f.read().strip().split('\n')
    lines = [line.split() for line in lines]
    for line in lines:
        e2m_dict[line[0]] = line[1:]
    return e2m_dict

def get_member2event_dict(num_members_total,e2m_dict,event_ids):
    m2e_dict = {}
    for i in range(num_members_total):
        m2e_dict[str(i)] = []
    for event_id in event_ids:
        members = e2m_dict[event_id]
        for member in members:
            m2e_dict[member].append(event_id)
    return m2e_dict


def get_meetup_biparticle_graph(config):
    train_events_ids, valid_events_ids, test_events_ids, num_events_total = get_event_train_valid_test_sets(config)
    num_members_total = get_member_info(config)

    e2m_dict = get_event2member_dict(config,num_members_total,num_events_total)
    m2e_dict_train = get_member2event_dict(num_members_total,e2m_dict,train_events_ids)
    num_zero_event_member = 0
    for key in m2e_dict_train.keys():
        if len(m2e_dict_train[key])==0:
            num_zero_event_member += 1
    print(num_zero_event_member)

    g = dgl.DGLGraph()
    g.add_nodes(num_events_total+num_members_total)
    # get train graph


    labels = [0] * len(lines_event) + [1]*len(lines_member)
    g.ndata['h'] = torch.nn.functional.one_hot(torch.tensor(labels))
    src,dst = [],[]
    for line in lines:
        temp = line.split()
        for t in temp[1:]:
            src.append(int(temp[0])-1)
            dst.append(int(t))
    g.add_edges(src,dst)
    g.edata['e'] = torch.tensor([1]*g.num_edges())
    return g