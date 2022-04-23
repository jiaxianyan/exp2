import argparse
import commons, dataset, model, runner
import torch
import itertools

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./config/default_config.yaml',help='path of dataset')
    args = parser.parse_args()

    config = commons.get_config_easydict(args.config_path)

    g_train_pos, g_train_neg, valid_events_ids, test_events_ids, e2m_dict, e2g_dict, g2m_dict, num_members_total = dataset.get_meetup_biparticle_graph(config)

    config.train.device = torch.device('cuda:' + str(config.train.gpuid) if torch.cuda.is_available() else "cpu")

    g_encoder = model.GraphSAGE(config).to(config.train.device)

    pred = model.MLPPredictor(config).to(config.train.device)

    optimizer = torch.optim.Adam(itertools.chain(g_encoder.parameters(), pred.parameters()), lr=0.01)

    solver = runner.DefaultRunner(g_train_pos, g_train_neg, valid_events_ids, test_events_ids, e2m_dict, e2g_dict, g2m_dict, num_members_total, g_encoder, pred, optimizer, config)

    solver.train()
