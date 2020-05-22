import torch
from torch import nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, T, decoder_num_hidden, encoder_num_hidden):
        super(Decoder, self).__init__()
        self.decoder_num_hidden = decoder_num_hidden
        self.encoder_num_hidden = encoder_num_hidden
        self.T = T
        self.attn_layer = nn.Sequential(nn.Linear(2 * decoder_num_hidden + encoder_num_hidden, encoder_num_hidden),
                                        nn.Tanh(), nn.Linear(encoder_num_hidden, 1))
        self.lstm_layer = nn.LSTM(input_size=1, hidden_size=decoder_num_hidden)
        self.fc = nn.Linear(encoder_num_hidden + 1, 1)
        self.fc_final_beta = nn.Linear(decoder_num_hidden + encoder_num_hidden, 1)
        self.fc_final_gamma = nn.Linear(decoder_num_hidden + encoder_num_hidden, 1)
        self.fc_final_sigma = nn.Linear(decoder_num_hidden + encoder_num_hidden, 1)

        self.fc.weight.data.normal_()

    def forward(self, X_encoed, y_prev):
        d_n = self._init_states(X_encoed)
        c_n = self._init_states(X_encoed)
        for t in range(self.T - 1):
            x = torch.cat((d_n.repeat(self.T - 1, 1, 1).permute(1, 0, 2),
                           c_n.repeat(self.T - 1, 1, 1).permute(1, 0, 2),
                           X_encoed), dim=2)
            beta = F.softmax(self.attn_layer(
                x.view(-1, 2 * self.decoder_num_hidden + self.encoder_num_hidden)).view(-1, self.T - 1), dim=1)
            # Eqn. 14: compute context vector
            # batch_size * encoder_hidden_size
            context = torch.bmm(beta.unsqueeze(1), X_encoed)[:, 0, :]
            if t < self.T - 1:
                # Eqn. 15
                # batch_size * 1
                y_tilde = self.fc(
                    torch.cat((context, y_prev[:, t].unsqueeze(1)), dim=1))
                # Eqn. 16: LSTM
                self.lstm_layer.flatten_parameters()
                _, final_states = self.lstm_layer(
                    y_tilde.unsqueeze(0), (d_n, c_n))
                # 1 * batch_size * decoder_num_hidden
                d_n = final_states[0]
                # 1 * batch_size * decoder_num_hidden
                c_n = final_states[1]
        # Eqn. 22: final output
        final_temp_y = torch.cat((d_n[0], context), dim=1)
        y_pred_beta = self.fc_final_beta(final_temp_y)
        y_pred_gamma = self.fc_final_gamma(final_temp_y)
        y_pred_sigma = self.fc_final_sigma(final_temp_y)
        return y_pred_beta, y_pred_sigma, y_pred_gamma

    def _init_states(self, X):
        initial_states = X.data.new(
            1, X.size(0), self.decoder_num_hidden).zero_()
        return initial_states
