import os
import dgl
import torch
import commons
from tqdm import tqdm
from time import time

class DefaultRunner(object):
    def __init__(self,graphs, e2m_dict, e2g_dict, g2m_dict, num_members_total, g_encoder, pred, optimizer, config):
        self.config = config
        self.device = config.train.device
        self.optimizer = optimizer

        self.g_train_pos = graphs[0].to(self.device)
        self.g_train_neg = graphs[1].to(self.device)
        self.g_valid_pos = graphs[2].to(self.device)
        self.g_valid_neg = graphs[3].to(self.device)
        self.g_test_pos = graphs[4].to(self.device)
        self.g_test_neg = graphs[5].to(self.device)

        self.e2m_dict = e2m_dict
        self.e2g_dict = e2g_dict
        self.g2m_dict = g2m_dict
        self.num_members_total = num_members_total
        self.num_events_total = len(e2m_dict.keys())

        self.g_encoder = g_encoder
        self.pred = pred

        self.best_f1 = 0
        self.start_epoch = 0
        self.best_matrics = []

    def save(self, checkpoint, epoch=None, var_list={}):
        state = {
            **var_list,
            "encoder": self.g_encoder.state_dict(),
            "pred": self.pred.state_dict(),
            "config": self.config
        }
        epoch = str(epoch) if epoch is not None else ''
        checkpoint = os.path.join(checkpoint, 'checkpoint%s' % epoch)
        torch.save(state, checkpoint)

    def load(self, checkpoint, epoch=None, load_optimizer=False, load_scheduler=False):

        epoch = str(epoch) if epoch is not None else ''
        checkpoint = os.path.join(checkpoint, 'checkpoint%s' % epoch)
        print("Load checkpoint from %s" % checkpoint)

        state = torch.load(checkpoint, map_location=self.device)
        self.g_encoder.load_state_dict(state["encoder"])
        # self._model.load_state_dict(state["model"], strict=False)
        self.best_loss = state['best_loss']
        self.start_epoch = state['cur_epoch'] + 1

        if load_optimizer:
            self.optimizer.load_state_dict(state["optimizer"])
            if self.device.type == 'cuda':
                for state in self._optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda(self.device)

        if load_scheduler:
            self.scheduler.load_state_dict(state["scheduler"])

    @torch.no_grad()
    def evaluate_train(self, split, h):
        test_pos_ids = getattr(self, f"g_{split}_pos")
        test_neg_ids = getattr(self, f"g_{split}_neg")
        pos_score = self.pred(test_pos_ids, h)
        neg_score = self.pred(test_neg_ids, h)

        auc, acc, recall, f1 = commons.compute_auc(pos_score, neg_score), commons.compute_acc(pos_score, neg_score), commons.compute_recall(pos_score, neg_score), commons.compute_f1(pos_score, neg_score)
        print('{} AUC : {}, acc : {}, recall : {}, F1 : {}'.format(split, auc, acc, recall, f1))

        return auc, acc, recall, f1

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
        train_start = time()
        start_epoch = self.start_epoch
        for epoch in range(self.config.train.epochs):
            h = self.g_encoder(self.g_train_pos, self.g_train_pos.ndata['h'])
            pos_score = self.pred(self.g_train_pos, h)
            neg_score = self.pred(self.g_train_neg, h)
            loss = commons.compute_loss(pos_score, neg_score, self.device)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if epoch % 5 == 0:
                print('In epoch {}, loss: {}'.format(epoch, loss))
                _ = self.evaluate_train('train', h)
                valid_matric = self.evaluate_train('valid', h)
                test_matric = self.evaluate_train('test', h)

            if self.best_f1 < valid_matric[-1]:
                self.best_f1 = valid_matric[-1]
                self.best_matrics = test_matric
                if self.config.train.save:
                    print('saving checkpoint')
                    val_list = {
                        'cur_epoch': epoch + start_epoch,
                        'best_loss': self.best_f1,
                    }
                    self.save(self.config.train.save_path, epoch + start_epoch, val_list)

        print('Total time elapsed: %.5fs' % (time() - train_start))

        return self.best_matrics