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
                 norm_rnn_hidden_size=32,
                 norm_rnn_layers=2,
                 kernel_size=[3, 3],
                 stride=[2, 2],
                 channels=[1, 16, 16, 16, 32]):
        super(Model, self).__init__()

        layers = []
        layers += [torch.nn.Linear(in_features=512, out_features=norm_rnn_hidden_size)]
        layers += [torch.nn.ELU()]
        layers += [RnnWrapper(torch.nn.LSTM(input_size=norm_rnn_hidden_size,
                                            hidden_size=norm_rnn_hidden_size,
                                            num_layers=norm_rnn_layers,
                                            batch_first=True))]
        layers += [torch.nn.Linear(in_features=norm_rnn_hidden_size, out_features=512*2)]
        self.norm_net = torch.nn.Sequential(*layers)

        self.cnns = torch.nn.ModuleList()
        for i in range(len(channels) - 1):
            layers = []
            layers += [torch.nn.Conv2d(in_channels=channels[i],
                                       out_channels=channels[i+1],
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=[0, 0])]
            layers += [torch.nn.BatchNorm2d(num_features=channels[i+1])]
            layers += [torch.nn.ELU()]
            # if max_pool is not None:
            #     layers += [torch.nn.MaxPool2d(kernel_size=max_pool)]
            # if dropout is not None:
            #     layers += [torch.nn.Dropout2d(p=dropout)]
            self.cnns.append(torch.nn.Sequential(*layers))

        self.ada_max_pool = torch.nn.AdaptiveMaxPool2d(output_size=[1, 1])
        self.fc_out = torch.nn.Linear(in_features=channels[-1], out_features=11)

    def forward(self, feature):
        mean, logstd = self.norm_net(feature).chunk(2, dim=-1)
        norm_feature = (feature - mean) / torch.exp(logstd)
        
        x = norm_feature.unsqueeze(1)
        for cnn in self.cnns:
            x = cnn(x)
        
        out = self.ada_max_pool(x).squeeze()
        out = self.fc_out(out)
        return out
    
    def total_parameter(self):
        return sum([p.numel() for p in self.parameters()])

if __name__ == '__main__':
    model = Model()
    print('%fM' % (model.total_parameter() / 1024 / 1024))
    feature = torch.ones(6, 157, 512, dtype=torch.float32)
    print(feature.shape)
    out = model(feature)
    print(out.shape)