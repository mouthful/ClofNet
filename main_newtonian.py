import os
import json
import time
import logging
import argparse

import torch
from torch import nn, optim
from newtonian.dataset4newton import NBodyDataset
from newtonian.gnn import GNN, RF_vel
from newtonian.egnn import EGNN, EGNN_vel
from newtonian.clof import ClofNet, ClofNet_vel, ClofNet_vel_gbf

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--exp_name', type=str, default='exp_1', metavar='N', help='experiment_name')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=5, metavar='N',
                    help='how many epochs to wait before logging test')
parser.add_argument('--outf', type=str, default='saved', metavar='N',
                    help='folder to output')
parser.add_argument('--data_mode', type=str, default='small', metavar='N',
                    help='folder to dataset')
parser.add_argument('--data_root', type=str, default='dataset/clofnet_dataset', metavar='N',
                    help='folder to dataset root')
parser.add_argument('--lr', type=float, default=5e-4, metavar='N',
                    help='learning rate')
parser.add_argument('--nf', type=int, default=64, metavar='N',
                    help='learning rate')
parser.add_argument('--model', type=str, default='egnn_vel', metavar='N',
                    help='available models: gnn, baseline, linear, linear_vel, se3_transformer, egnn_vel, rf_vel, tfn')
parser.add_argument('--attention', type=int, default=0, metavar='N',
                    help='attention in the ae model')
parser.add_argument('--n_layers', type=int, default=4, metavar='N',
                    help='number of layers for the autoencoder')
parser.add_argument('--degree', type=int, default=2, metavar='N',
                    help='degree of the TFN and SE3')
parser.add_argument('--max_training_samples', type=int, default=3000, metavar='N',
                    help='maximum amount of training samples')
parser.add_argument('--sweep_training', type=int, default=0, metavar='N',
                    help='0 nor sweep, 1 sweep, 2 sweep small')
parser.add_argument('--time_exp', type=int, default=0, metavar='N',
                    help='timing experiment')
parser.add_argument('--weight_decay', type=float, default=1e-12, metavar='N',
                    help='timing experiment')
parser.add_argument('--div', type=float, default=1, metavar='N',
                    help='timing experiment')
parser.add_argument('--norm_diff', type=eval, default=False, metavar='N',
                    help='normalize_diff')
parser.add_argument('--tanh', type=eval, default=False, metavar='N',
                    help='use tanh')
parser.add_argument('--LR_decay', type=eval, default=False, metavar='N',
                    help='LR_decay')
parser.add_argument('--decay', type=float, default=0.1, metavar='N',
                    help='learning rate decay')

time_exp_dic = {'time': 0, 'counter': 0}
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device("cuda" if args.cuda else "cpu")
loss_mse = nn.MSELoss()

print(args)
try:
    os.makedirs(args.outf)
except OSError:
    pass

try:
    os.makedirs(args.outf + "/" + args.exp_name)
except OSError:
    pass

# prepare data root and save path for checkpoint
data_root = os.path.join(args.data_root, args.data_mode)
checkpoint_path = os.path.join(args.outf, args.exp_name, 'checkpoint')
os.makedirs(checkpoint_path, exist_ok=True)

def get_velocity_attr(loc, vel, rows, cols):

    diff = loc[cols] - loc[rows]
    norm = torch.norm(diff, p=2, dim=1).unsqueeze(1)
    u = diff/norm
    va, vb = vel[rows] * u, vel[cols] * u
    va, vb = torch.sum(va, dim=1).unsqueeze(1), torch.sum(vb, dim=1).unsqueeze(1)
    return va

def main():
    logging.basicConfig(
        filename=os.path.join(args.outf, args.exp_name, "training.log"),
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode='w',
        level=logging.INFO,
    )
    logging.info(f'load data from {data_root}')
    logging.info(f'save checkpoints to {checkpoint_path}')

    dataset_train = NBodyDataset(partition='train', max_samples=args.max_training_samples, data_root=data_root, data_mode=args.data_mode)
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True)

    dataset_val = NBodyDataset(partition='valid', data_root=data_root, data_mode=args.data_mode)
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, drop_last=False)

    dataset_test = NBodyDataset(partition='test', data_root=data_root, data_mode=args.data_mode)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=False)


    if args.model == 'gnn':
        model = GNN(input_dim=6, hidden_nf=args.nf, n_layers=args.n_layers, device=device, recurrent=True)
    elif args.model == 'egnn':
        model = EGNN(in_node_nf=1, in_edge_nf=2, hidden_nf=args.nf, device='cpu', n_layers=args.n_layers)
    elif args.model == 'egnn_vel':
        model = EGNN_vel(in_node_nf=1, in_edge_nf=2, hidden_nf=args.nf, device=device, n_layers=args.n_layers, recurrent=True, norm_diff=args.norm_diff, tanh=args.tanh)
    elif args.model == 'rf_vel':
        model = RF_vel(hidden_nf=args.nf, edge_attr_nf=2, device=device, act_fn=nn.SiLU(), n_layers=args.n_layers)
    elif args.model == 'clof':
        model = ClofNet(in_node_nf=1, in_edge_nf=2, hidden_nf=args.nf, n_layers=args.n_layers, device=device, recurrent=True, norm_diff=args.norm_diff, tanh=args.tanh)
    elif args.model == 'clof_vel':
        model = ClofNet_vel(in_node_nf=1, in_edge_nf=2, hidden_nf=args.nf, n_layers=args.n_layers, device=device, recurrent=True, norm_diff=args.norm_diff, tanh=args.tanh)
    elif args.model == 'clof_vel_gbf':
        model = ClofNet_vel_gbf(in_node_nf=1, in_edge_nf=2, hidden_nf=args.nf, n_layers=args.n_layers, device=device, recurrent=True, norm_diff=args.norm_diff, tanh=args.tanh)
    else:
        raise Exception("Wrong model specified")

    logging.info(args)
    print(model)
    logging.info(model)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    step_size = int(args.epochs // 8)
    if args.LR_decay:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=args.decay, last_epoch=-1)

    results = {'epochs': [], 'losess': []}
    best_val_loss = 1e8
    best_test_loss = 1e8
    best_epoch = 0
    for epoch in range(0, args.epochs):
        loss = train(model, optimizer, epoch, loader_train)
        if args.LR_decay:
            scheduler.step()
        if epoch % args.test_interval == 0:
            val_loss = train(model, optimizer, epoch, loader_val, backprop=False)
            test_loss = train(model, optimizer, epoch, loader_test, backprop=False)
            results['epochs'].append(epoch)
            results['losess'].append(test_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_test_loss = test_loss
                best_epoch = epoch
                torch.save(model, os.path.join(checkpoint_path, 'best_model.pt'))
            print("*** Best Val Loss: %.5f \t Best Test Loss: %.5f \t Best epoch %d" % (best_val_loss, best_test_loss, best_epoch))
            logging.info("*** Best Val Loss: %.5f \t Best Test Loss: %.5f \t Best epoch %d" % (best_val_loss, best_test_loss, best_epoch))

        json_object = json.dumps(results, indent=4)
        with open(args.outf + "/" + args.exp_name + "/losess.json", "w") as outfile:
            outfile.write(json_object)
    return best_val_loss, best_test_loss, best_epoch


def train(model, optimizer, epoch, loader, backprop=True):
    if backprop:
        model.train()
    else:
        model.eval()

    res = {'epoch': epoch, 'loss': 0, 'coord_reg': 0, 'counter': 0}

    for batch_idx, data in enumerate(loader):
        batch_size, n_nodes, _ = data[0].size()
        data = [d.to(device) for d in data]
        data = [d.view(-1, d.size(2)) for d in data]
        loc, vel, edge_attr, charges, loc_end = data

        edges = loader.dataset.get_edges(batch_size, n_nodes)
        edges = [edges[0].to(device), edges[1].to(device)]

        optimizer.zero_grad()

        if args.time_exp:
            torch.cuda.synchronize()
            t1 = time.time()

        if args.model == 'gnn':
            nodes = torch.cat([loc, vel], dim=1)
            loc_pred = model(nodes, edges, edge_attr)
        elif args.model == 'egnn':
            nodes = torch.ones(loc.size(0), 1).to(device)  # all input nodes are set to 1
            rows, cols = edges
            loc_dist = torch.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1)  # relative distances among locations
            vel_attr = get_velocity_attr(loc, vel, rows, cols).detach()
            edge_attr = torch.cat([edge_attr, loc_dist, vel_attr], 1).detach()  # concatenate all edge properties
            loc_pred = model(nodes, loc.detach(), edges, edge_attr)
        elif args.model == 'egnn_vel':
            nodes = torch.sqrt(torch.sum(vel ** 2, dim=1)).unsqueeze(1).detach()
            rows, cols = edges
            loc_dist = torch.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1)  # relative distances among locations
            edge_attr = torch.cat([edge_attr, loc_dist], 1).detach()  # concatenate all edge properties
            loc_pred = model(nodes, loc.detach(), edges, vel, edge_attr)
        elif args.model == 'rf_vel':
            rows, cols = edges
            vel_norm = torch.sqrt(torch.sum(vel ** 2, dim=1).unsqueeze(1)).detach()
            loc_dist = torch.sum((loc[rows] - loc[cols]) ** 2, 1).unsqueeze(1)
            edge_attr = torch.cat([edge_attr, loc_dist], 1).detach()
            loc_pred = model(vel_norm, loc.detach(), edges, vel, edge_attr)
        elif args.model in ['clof', 'clof_vel', 'clof_vel_gbf']:
            nodes = torch.sqrt(torch.sum(vel ** 2, dim=1)).unsqueeze(1).detach()
            rows, cols = edges
            loc_dist = torch.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1)  # relative distances among locations
            edge_attr = torch.cat([edge_attr, loc_dist], 1).detach()  # concatenate all edge properties
            loc_pred = model(nodes, loc.detach(), edges, vel, edge_attr, n_nodes=n_nodes)
        else:
            raise Exception("Wrong model")

        if args.time_exp:
            torch.cuda.synchronize()
            t2 = time.time()
            time_exp_dic['time'] += t2 - t1
            time_exp_dic['counter'] += 1

            print("Forward average time: %.6f" % (time_exp_dic['time'] / time_exp_dic['counter']))
            logging.info("Forward average time: %.6f" % (time_exp_dic['time'] / time_exp_dic['counter']))
        loss = loss_mse(loc_pred, loc_end)
        if backprop:
            loss.backward()
            optimizer.step()
        res['loss'] += loss.item()*batch_size
        res['counter'] += batch_size
        if batch_idx % args.log_interval == 0 and (args.model == "se3_transformer" or args.model == "tfn"):
            print('===> {} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(loader.dataset.partition,
                epoch, batch_idx * batch_size, len(loader.dataset),
                100. * batch_idx / len(loader),
                loss.item()))
            logging.info('===> {} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(loader.dataset.partition,
                epoch, batch_idx * batch_size, len(loader.dataset),
                100. * batch_idx / len(loader),
                loss.item()))
    if not backprop:
        prefix = "==> "
    else:
        prefix = ""
    print('%s epoch %d avg loss: %.5f LR: %.6f' % (prefix+loader.dataset.partition, epoch, res['loss'] / res['counter'], optimizer.param_groups[0]['lr']))
    logging.info('%s epoch %d avg loss: %.5f LR: %.6f' % (prefix+loader.dataset.partition, epoch, res['loss'] / res['counter'], optimizer.param_groups[0]['lr']))

    return res['loss'] / res['counter']


def main_sweep():
    training_samples = [200, 400, 800, 1600, 3200, 6400, 12800, 25000, 50000]
    n_epochs = [200, 200, 200, 200, 500, 500, 600, 600, 600]

    if args.sweep_training == 2:
        training_samples = training_samples[0:5]
        n_epochs = n_epochs[0:5]
    elif args.sweep_training == 3:
        training_samples = training_samples[6:]
        n_epochs = n_epochs[6:]
    elif args.sweep_training == 4:
        training_samples = training_samples[8:]
        n_epochs = n_epochs[8:]


    results = {'tr_samples': [], 'test_loss': [], 'best_epochs': []}
    for epochs, tr_samples in zip(n_epochs, training_samples):
        args.epochs = epochs
        args.max_training_samples = tr_samples
        args.test_interval = max(int(10000/tr_samples), 1)
        best_val_loss, best_test_loss, best_epoch = main()
        results['tr_samples'].append(tr_samples)
        results['best_epochs'].append(best_epoch)
        results['test_loss'].append(best_test_loss)
        print("\n####### Results #######")
        print(results)
        print("Results for %d epochs and %d # training samples \n" % (epochs, tr_samples))
        logging.info("\n####### Results #######")
        logging.info(results)
        logging.info("Results for %d epochs and %d # training samples \n" % (epochs, tr_samples))


if __name__ == "__main__":
    if args.sweep_training:
        main_sweep()
    else:
        main()




