import argparse
from ops import read_data, cut_day
from torch.autograd import Variable
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
import time
import pandas as pd
from Model.Encoder import Encoder
from Model.Decoder import Decoder
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--encoder_num_hidden', default=512, type=int)
parser.add_argument('--decoder_num_hidden', default=512, type=int)
parser.add_argument('--learning_rate', default=1e-2, type=float)
parser.add_argument('--batch_size', default=7, type=float)
parser.add_argument('--epochs', default=20, type=int)
parser.add_argument('--parallel', default=False, type=bool)
parser.add_argument('--T', default=14, type=int)
parser.add_argument('--seed', default=1234, type=int)
parser.add_argument('--load', default=False, type=bool)  # load model
parser.add_argument('--print_log', default=10, type=int)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tnow = time.strftime('%m%d', time.localtime(time.time()))
directory = './Output/' + tnow + '_' + str(args.learning_rate) + '_' + str(args.epochs) + '/'
Train_loss = {'beta_loss': [], 'sigma_loss': [], 'gamma_loss': []}
Test_loss = {'beta_loss': [], 'sigma_loss': [], 'gamma_loss': []}


def get_accuracy(truth, pred):
    assert len(truth) == len(pred)
    return F.mse_loss(truth.squeeze(1), pred)


class DA_rnn(nn.Module):
    def __init__(self, train_data, test_data):
        super(DA_rnn, self).__init__()
        X, beta, sigma, gamma = train_data
        self.test_data = test_data
        self.shuffle = False
        self.X = X
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma
        self.writer = SummaryWriter(directory)
        self.Encoder = Encoder(input_size=X.shape[1], encoder_num_hidden=args.encoder_num_hidden, T=args.T).to(device)
        self.Decoder = Decoder(encoder_num_hidden=args.encoder_num_hidden, decoder_num_hidden=args.decoder_num_hidden,
                               T=args.T).to(device)
        # Loss function
        self.criterion_beta = nn.MSELoss()
        self.criterion_sigma = nn.MSELoss()
        self.criterion_gamma = nn.MSELoss()
        if args.parallel:
            self.Encoder = nn.DataParallel(self.Encoder)
            self.Decoder = nn.DataParallel(self.Decoder)
        self.encoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad,
                                                          self.Encoder.parameters()),
                                            lr=args.learning_rate)
        self.decoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad,
                                                          self.Decoder.parameters()),
                                            lr=args.learning_rate)
        self.train_timesteps = int(self.X[:cut_day].shape[0])
        self.input_size = self.X.shape[1]
        self.num_of_test = 0

    def train(self):
        iter_per_epoch = int(np.ceil(self.train_timesteps * 1. / args.batch_size))
        self.iter_losses = np.zeros(args.epochs * iter_per_epoch)
        self.epoch_losses = np.zeros(args.epochs)
        n_iter = 0
        for epoch in range(args.epochs):
            if self.shuffle:
                ref_idx = np.random.permutation(self.train_timesteps - args.T)
            else:
                ref_idx = np.array(range(self.train_timesteps - args.T))
            idx = 0
            run_loss_beta, run_loss_sigma, run_loss_gamma = 0, 0, 0
            while (idx < self.train_timesteps - args.T):
                indices = ref_idx[idx:(idx + args.batch_size)]
                x = np.zeros((len(indices), args.T - 1, self.input_size))
                beta_prev = np.zeros((len(indices), args.T - 1))
                beta_gt = self.beta[indices + args.T]
                sigma_gt = self.sigma[indices + args.T]
                gamma_gt = self.gamma[indices + args.T]
                for bs in range(len(indices)):
                    x[bs, :, :] = self.X[indices[bs]:(indices[bs] + args.T - 1), :]
                    beta_prev[bs, :] = self.beta[indices[bs]:(indices[bs] + args.T - 1)]
                loss_beta, loss_sigma, loss_gamma, acc_sigma, acc_gamma = self.train_forward(x, beta_prev, beta_gt,
                                                                                             sigma_gt, gamma_gt)
                run_loss_beta += loss_beta.item()
                run_loss_sigma += loss_sigma.item()
                run_loss_gamma += loss_gamma.item()
                loss = loss_beta + loss_sigma + loss_gamma
                self.iter_losses[epoch * iter_per_epoch + idx // args.batch_size] = loss.item()
                idx += args.batch_size
                n_iter += 1
                if n_iter % 4000 == 0 and n_iter != 0:
                    for param_group in self.encoder_optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.8
                    for param_group in self.decoder_optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.8
                self.epoch_losses[epoch] = np.mean(
                    self.iter_losses[range(epoch * iter_per_epoch, (epoch + 1) * iter_per_epoch)])
            self.writer.add_scalar('loss/loss_beta', run_loss_beta, global_step=epoch)
            self.writer.add_scalar('loss/loss_sigma', run_loss_sigma, global_step=epoch)
            self.writer.add_scalar('loss/loss_gamma', run_loss_gamma, global_step=epoch)
            Train_loss['beta_loss'].append(run_loss_beta)
            Train_loss['sigma_loss'].append(run_loss_sigma)
            Train_loss['gamma_loss'].append(run_loss_gamma)
            if epoch % args.print_log == 0:
                test_loss = self.test()
                print("Epochs:{}, Iterations:{}, Train_loss:{}, Test_loss:{}".format(epoch, n_iter,
                                                                                     round(self.epoch_losses[epoch], 3),
                                                                                     round(test_loss, 3)))

    def train_forward(self, X, beta_prev, beta_gt, sigma_gt, gamma_gt):
        # zero gradients
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        input_weighted, input_encoded = self.Encoder(
            Variable(torch.from_numpy(X).type(torch.FloatTensor).to(device)))
        y_pred_beta, y_pred_sigma, y_pred_gamma = self.Decoder(input_encoded, Variable(
            torch.from_numpy(beta_prev).type(torch.FloatTensor).to(device)))
        # print(y_pred_sigma)
        y_true_beta = torch.from_numpy(beta_gt).type(torch.FloatTensor).to(device)
        y_true_beta = y_true_beta.view(-1, 1)
        y_true_sigma = torch.from_numpy(sigma_gt).type(torch.FloatTensor).unsqueeze(1).to(device)
        y_true_gamma = torch.from_numpy(gamma_gt).type(torch.FloatTensor).unsqueeze(1).to(device)
        y_sigma_dev = torch.max(y_pred_sigma, 1)[1].type(torch.FloatTensor).to(device)
        y_gamma_dev = torch.max(y_pred_gamma, 1)[1].type(torch.FloatTensor).to(device)
        acc_sigma = get_accuracy(y_true_sigma, y_sigma_dev)
        acc_gamma = get_accuracy(y_true_gamma, y_gamma_dev)
        loss_beta = self.criterion_beta(y_pred_beta, y_true_beta)
        loss_sigma = self.criterion_sigma(y_pred_sigma, y_true_sigma)
        loss_gamma = self.criterion_gamma(y_pred_gamma, y_true_gamma)
        loss_beta.backward(retain_graph=True)
        loss_sigma.backward(retain_graph=True)
        loss_gamma.backward(retain_graph=True)
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        return loss_beta, loss_sigma, loss_gamma, acc_sigma, acc_gamma

    def test(self):
        self.test_X, self.test_beta, self.test_sigma, self.test_gamma = self.test_data
        y_pred_beta = np.zeros(self.train_timesteps - args.T + 1)
        # y_pred_gamma = np.zeros(self.train_timesteps - args.T + 1)
        # y_pred_sigma = np.zeros(self.train_timesteps - args.T + 1)
        i = 0
        run_loss_beta, run_loss_sigma, run_loss_gamma = 0, 0, 0
        while i < len(y_pred_beta):
            batch_idx = np.array(range(len(y_pred_beta)))[i: (i + args.batch_size)]
            X = np.zeros((len(batch_idx), args.T - 1, self.X.shape[1]))
            y_history = np.zeros((len(batch_idx), args.T - 1))
            for j in range(len(batch_idx)):
                X[j, :, :] = self.test_X[range(batch_idx[j], batch_idx[j] + args.T - 1), :]
                y_history[j, :] = self.test_beta[range(batch_idx[j], batch_idx[j] + args.T - 1)]
            beta_gt = self.test_beta[batch_idx + args.T]
            sigma_gt = self.test_sigma[batch_idx + args.T]
            gamma_gt = self.test_gamma[batch_idx + args.T]
            y_true_beta = torch.from_numpy(beta_gt).type(torch.FloatTensor).to(device)
            y_true_beta = y_true_beta.view(-1, 1)
            y_true_sigma = torch.from_numpy(sigma_gt).type(torch.FloatTensor).unsqueeze(1).to(device)
            y_true_gamma = torch.from_numpy(gamma_gt).type(torch.FloatTensor).unsqueeze(1).to(device)
            y_history = Variable(torch.from_numpy(y_history).type(torch.FloatTensor).to(device))
            _, input_encoded = self.Encoder(Variable(torch.from_numpy(X).type(torch.FloatTensor).to(device)))
            y_pred_beta, y_pred_sigma, y_pred_gamma = self.Decoder(input_encoded, y_history)
            loss_beta = self.criterion_beta(y_pred_beta, y_true_beta)
            loss_sigma = self.criterion_sigma(y_pred_sigma, y_true_sigma)
            loss_gamma = self.criterion_gamma(y_pred_gamma, y_true_gamma)
            run_loss_beta += loss_beta.item()
            run_loss_sigma += loss_sigma.item()
            run_loss_gamma += loss_gamma.item()
            # y_pred_beta = y_pred_beta[i:(i + args.batch_size)]
            # y_pred_beta = y_pred_beta.cpu().detach().numpy()[:, 0]
            # y_pred_sigma = y_pred_sigma[i:(i + args.batch_size)]
            # y_pred_gamma = y_pred_gamma[i:(i + args.batch_size)]
            i += args.batch_size
        self.writer.add_scalar('test_loss/loss_beta', run_loss_beta, global_step=self.num_of_test)
        self.writer.add_scalar('test_loss/loss_sigma', run_loss_sigma, global_step=self.num_of_test)
        self.writer.add_scalar('test_loss/loss_gamma', run_loss_gamma, global_step=self.num_of_test)
        Test_loss['beta_loss'].append(run_loss_beta)
        Test_loss['sigma_loss'].append(run_loss_sigma)
        Test_loss['gamma_loss'].append(run_loss_gamma)
        self.num_of_test += 1
        test_loss = run_loss_beta + run_loss_sigma + run_loss_gamma
        return test_loss
        # return y_pred_beta, torch.max(y_pred_sigma, 1)[1], torch.max(y_pred_gamma, 1)[1]


def model_save(model):
    torch.save(model, './model_save/model_' + tnow + '_' + str(args.learning_rate) + '_' + str(args.epochs) + '.pkl')


def main():
    train_data = read_data("data/data_used/covid_data_cn.csv", "data/data_used/search_cn.csv")
    test_data = read_data("data/data_used/covid_data_us.csv", "data/data_used/search_us.csv")
    model = DA_rnn(train_data, test_data)
    model.train()
    Train_loss_result = pd.DataFrame(Train_loss)
    Train_loss_result.to_csv(
        './Result/train_loss' + tnow + '_' + str(args.learning_rate) + '_' + str(args.epochs) + '.csv',
        index=None)
    Test_loss_result = pd.DataFrame(Train_loss)
    Test_loss_result.to_csv(
        './Result/test_loss' + tnow + '_' + str(args.learning_rate) + '_' + str(args.epochs) + '.csv',
        index=None)
    # model_save(model)
    model.test()


if __name__ == '__main__':
    main()
