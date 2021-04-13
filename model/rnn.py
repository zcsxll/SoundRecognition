import torch

class RnnWrapper(torch.nn.Module):
    def __init__(self, rnn):
        super(RnnWrapper, self).__init__()
        self.rnn = rnn

    def forward(self, x, h=None, c=None):
        if callable(getattr(self.rnn, 'flatten_parameters', None)):
            self.rnn.flatten_parameters()
        if h is None or c is None:
            return self.rnn(x)[0]
        # print(h.shape, c.shape)
        return self.rnn(x, (h, c))

class Model(torch.nn.Module):
    def __init__(self,
                rnn_type='LSTM',
                norm_rnn_hidden_size=32,
                norm_rnn_layers=2,
                rnn_hidden_size=128,
                rnn_layers=4,
                rnn_bidirectional=True):
        super(Model, self).__init__()

        if rnn_type == 'GRU':
            RNN = torch.nn.GRU
        elif rnn_type == 'LSTM':
            RNN = torch.nn.LSTM
        else:
            raise NotImplementedError

        layers = []
        layers += [torch.nn.Linear(in_features=512, out_features=norm_rnn_hidden_size)]
        layers += [torch.nn.ELU()]
        layers += [RnnWrapper(RNN(input_size=norm_rnn_hidden_size,
                                hidden_size=norm_rnn_hidden_size,
                                num_layers=norm_rnn_layers,
                                batch_first=True))]
        layers += [torch.nn.Linear(in_features=norm_rnn_hidden_size, out_features=512*2)]
        self.norm_net = torch.nn.Sequential(*layers)

        layers = []
        layers += [torch.nn.Linear(in_features=512, out_features=rnn_hidden_size)]
        layers += [torch.nn.ELU()]
        layers += [RnnWrapper(RNN(input_size=rnn_hidden_size,
                                hidden_size=rnn_hidden_size,
                                num_layers=rnn_layers,
                                batch_first=True,
                                bidirectional=rnn_bidirectional))]
        self.core_net = torch.nn.Sequential(*layers)

        if rnn_bidirectional:
            core_size = rnn_hidden_size * 2
        else:
            core_size = rnn_hidden_size
        self.out1 = torch.nn.Linear(in_features=core_size, out_features=1)
        self.elu1 = torch.nn.ELU()
        self.out2 = torch.nn.Linear(in_features=155, out_features=7)

    def forward(self, feature):
        mean, logstd = self.norm_net(feature).chunk(2, dim=-1)
        norm_feature = (feature - mean) / torch.exp(logstd)
        core_out = self.core_net(norm_feature)
        out1 = self.out1(core_out).squeeze(-1)
        out1 = self.elu1(out1)
        out2 = self.out2(out1).squeeze(-1)
        return out2

    def total_parameter(self):
        return sum([p.numel() for p in self.parameters()])

if __name__ == '__main__':
    model = Model()
    print('%fM' % (model.total_parameter() / 1024 / 1024))
    feature = torch.ones(6, 157, 512, dtype=torch.float32)
    print(feature.shape)
    out = model(feature)
    print(out.shape)