import dgl
import torch
import commons
from tqdm import tqdm

class DefaultRunner(object):
    def __init__(self,g_train_pos, g_train_neg, valid_events_ids, test_events_ids, e2m_dict, e2g_dict, g2m_dict, num_members_total, g_encoder, pred, optimizer, config):
        self.config = config
        self.device = config.train.device
        self.optimizer = optimizer

        self.g_train_pos = g_train_pos.to(self.device)
        self.g_train_neg = g_train_neg.to(self.device)

        self.valid_events_ids = valid_events_ids
        self.test_events_ids = test_events_ids

        self.e2m_dict = e2m_dict
        self.e2g_dict = e2g_dict
        self.g2m_dict = g2m_dict
        self.num_members_total = num_members_total
        self.num_events_total = len(e2m_dict.keys())

        self.g_encoder = g_encoder
        self.pred = pred

        self.best = 0

    @torch.no_grad()
    def evaluate_train(self, h):
        pos_score = self.pred(self.g_train_pos, h)
        neg_score = self.pred(self.g_train_neg, h)
        print('AUC', commons.compute_auc(pos_score, neg_score))

    @torch.no_grad()
    def evaluate_future(self,split,h):
        test_ids = getattr(self, f"{split}_events_ids")
        accs, recalls = [],[]
        for id in tqdm(test_ids):
            src = [id] * self.num_members_total
            dst = [i for i in range(self.num_members_total)]

            g = dgl.DGLGraph()
            g.add_nodes(self.g_train_neg.number_of_nodes())
            g.add_edges(src, dst)

            g = g.to(self.device)

            score = self.pred(g, h)
            positive_indexs = [int(index) for index in self.e2m_dict[id]]
            ground_labels = torch.zeros(len(score))
            ground_labels[positive_indexs] = 1
            acc = commons.accuracy(ground_labels,score.squeeze(),self.device)
            accs.append(acc)
            if len(positive_indexs)>0:
                recall = commons.recall(positive_indexs,score.squeeze())
                recalls.append(recall)
            else:
                recall = 'no members'

            print(f'event id :{id}, acc:{acc}, recall:{recall}')
        print(f'average acc : {sum(accs)/len(accs)}, average recall : {sum(recalls)/len(recalls)}')
        return sum(accs)/len(accs), sum(recalls)/len(recalls)

    def train(self):
        print('begin training ...')
        for i in range(self.config.train.epochs):
            h = self.g_encoder(self.g_train_pos, self.g_train_pos.ndata['h'])
            pos_score = self.pred(self.g_train_pos, h)
            neg_score = self.pred(self.g_train_neg, h)
            loss = commons.compute_loss(pos_score, neg_score, self.device)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i % 5 == 0:
                print('In epoch {}, loss: {}'.format(i, loss))
                self.evaluate_train(h)
        valid_acc, valid_recall = self.evaluate_future('valid',h)
        test_acc, test_recall = self.evaluate_future('test',h)
        print(f'valid_acc : {valid_acc}, valid_recall : {valid_recall}')
        print(f'test_acc : {test_acc}, test_recall : {test_recall}')
