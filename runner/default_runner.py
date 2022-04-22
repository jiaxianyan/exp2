import torch
import commons
class DefaultRunner(object):
    def __init__(self,g_train_pos, g_train_neg, valid_events_ids, test_events_ids, g_encoder, pred, optimizer, config):
        self.config = config
        self.optimizer = optimizer

        self.g_train_pos = g_train_pos
        self.g_train_neg = g_train_neg

        self.valid_events_ids = valid_events_ids
        self.test_events_ids = test_events_ids

        self.g_encoder = g_encoder
        self.pred = pred

    @torch.no_grad()
    def evaluate(self):
        pos_score = self.pred(self.g_train_pos, h)
        neg_score = self.pred(self.g_train_neg, h)
        print('AUC', commons.compute_auc(pos_score, neg_score))
        
        return

    def train(self):
        print('begin training ...')
        for i in range(self.config.train.epochs):
            h = self.g_encoder(self.g_train_pos, self.g_train_pos.ndata['h'])
            pos_score = self.pred(self.g_train_pos, h)
            neg_score = self.pred(self.g_train_neg, h)
            loss = commons.compute_loss(pos_score, neg_score)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i % 5 == 0:
                print('In epoch {}, loss: {}'.format(i, loss))


