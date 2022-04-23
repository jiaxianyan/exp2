import argparse
import commons, dataset, model, runner
import torch
import itertools

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./config/dynamics_config.yaml',help='path of dataset')
    args = parser.parse_args()

    config = commons.get_config_easydict(args.config_path)

    graphs_sequence, e2m_dict, e2g_dict, g2m_dict, num_members_total = dataset.get_meetup_biparticle_graph_sequence(config)

    config.train.device = torch.device('cuda:' + str(config.train.gpuid) if torch.cuda.is_available() else "cpu")

    matrics_sequence = []
    for index,graphs in enumerate(graphs_sequence):
        print('===============================================')
        print(f'begin No.{index+1} training.')

        g_encoder = model.GraphSAGE(config).to(config.train.device)

        pred = model.MLPPredictor(config).to(config.train.device)

        optimizer = torch.optim.Adam(itertools.chain(g_encoder.parameters(), pred.parameters()), lr=0.01)

        solver = runner.DefaultRunner(graphs, e2m_dict, e2g_dict, g2m_dict, num_members_total, g_encoder, pred, optimizer, config)

        matrics = solver.train()
        print('===============================================')

        matrics_sequence.append(matrics)
        commons.output_results_sequence(matrics_sequence)
