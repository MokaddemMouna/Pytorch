import numpy as np
import torch
from torch.autograd import Variable
from torch import nn, optim
import math
import random

from matplotlib import pyplot as plt


lines = np.load('./descrip-test/data/strokes-py3.npy',
                allow_pickle=True)

def calculate_example_proba(lines):
    ratios = [int(line.shape[0]/300) for line in lines if line.shape[0]>= 300]
    nb_generated_sequence = sum(ratios)
    probas = [ratio/nb_generated_sequence for ratio in ratios]
    return probas

def generate_samples(lines,batch_size):
    sample_list = []
    probas = calculate_example_proba(lines)
    for line in np.random.choice(a=lines,size=batch_size,replace=False,
                                 p=probas):
        seq_length = line.shape[0]
        if seq_length >= 300:
            if seq_length == 300:
                sample = line
            elif seq_length == 301:
                index = np.random.random_integers(0, 1)
                sample = line[index:index + 300]
            else:
                index = np.random.randint(seq_length-301)
                sample  = line[index:index+300]
            sample_list.append(sample)
    return sample_list


def batch_generator(seq_size=300,batch_size=50):
    cache = []
    while True:
        if len(cache) < batch_size:
            cache += generate_samples(lines,1000)
        else:
            x = torch.Tensor(cache[:batch_size]) \
                .transpose(0, 1) \
                .contiguous()
            cache = cache[batch_size:]
            yield Variable(x)

def plot_points(data):
    plt.figure(figsize=[16,4])
    plt.gca().invert_yaxis()
    plt.axis('equal')
    pts = np.array(data).cumsum(axis=0)
    data[-1][-1] = 1
    idx = [i for i, v in enumerate(data) if data[i][-1]==1]
    start = 0
    for end in idx:
        tmp = pts[start:end+1]
        plt.plot(tmp[:,0], tmp[:,1], linewidth=2)
        start = end+1

# g = batch_generator()
# one_batch = next(g)
# print('m')

# This function is used to make logarithm of small values stable
# (not infinitely negative anymore)
def logsumexp(x):
    x_max, _ = x.max(dim=1,keepdim=True)
    x_max_expand = x_max.expand(x.size())
    res =  x_max + torch.log((x-x_max_expand).exp().sum(dim=1, keepdim=True))
    return res

class GaussianHandWriting(nn.Module):
    def __init__(self,n_gaussian=20,dropout=0,rnn_size=256):
        super(GaussianHandWriting, self).__init__()
        self.n_gaussian = n_gaussian
        self.dropout = dropout
        self.rnn_size = rnn_size

        self.output = n_gaussian * 6 + 1
        self.rnn = nn.GRU(input_size=3,hidden_size=rnn_size,num_layers=2,
                          dropout=dropout)
        self.linear = nn.Linear(in_features=self.rnn_size,out_features=self.output)


    def forward(self, input, hidden=None):
        output, hidden = self.rnn(input=input)
        output = output.view(-1,self.rnn_size)
        output = self.linear(input=output)
        mu1,mu2,log_sigma1,log_sigma2,rho,pi,zo = output.split(split_size=self.n_gaussian,dim=1)
        return mu1,mu2,log_sigma1,log_sigma2,rho,pi,zo,hidden

    def calculate_loss(self,x_input,x_next,hidden=None):
        mu1, mu2, log_sigma1, log_sigma2, rho, pi, zo, hidden = self.forward(x_input,hidden)
        x_next = x_next.view(-1,3)
        x_lift,x1,x2 = x_next.split(1,dim=1)
        loss1 = self.logP_gaussian(x1,x2,mu1,mu2,log_sigma1,log_sigma2,rho,pi)
        # the loss is summed across rows and size_average=False eq to sum
        # without averaging
        loss2 = nn.functional.binary_cross_entropy_with_logits(input=zo,target=x_lift,size_average=False)
        loss = (loss1 + loss2)/x_input.shape[1] #average over mini-batch
        return loss, hidden

    def logP_gaussian(self,x1,x2,mu1, mu2, log_sigma1, log_sigma2, rho, pi):
        x1,x2 = x1.repeat(1,self.n_gaussian),x2.repeat(1,self.n_gaussian)
        rho = nn.functional.tanh(rho)
        sigma1,sigma2 = log_sigma1.exp(),log_sigma2.exp()
        log_pi = nn.functional.log_softmax(pi,dim=1)
        z1 = (x1 - mu1)/sigma1
        z2 = (x2 - mu2)/sigma2
        z = z1**2 + z2**2 - 2*rho*z1*z2
        log_gaussian = - math.log(2*math.pi) - log_sigma1 - log_sigma2 - 0.5*(1-rho**2).log()
        log_gaussian += -z/2/(1-rho**2)
        log_gaussian = logsumexp(log_gaussian+log_pi)
        return log_gaussian.sum()

    def draw_one_sample(self, mu1, mu2, log_sigma1, log_sigma2, rho, pi_logits, z0_logits):
        # draw Gaussian mixture
        pi = nn.functional.softmax(pi_logits)
        idx, = random.choices(range(self.n_gaussian), weights = pi.data.tolist()[0])
        sigma1, sigma2 = log_sigma1.exp(), log_sigma2.exp()
        x1 = torch.normal(mu1[:,idx],sigma1[:,idx]) # size = 1,
        mu2 = mu2+rho*(log_sigma2-log_sigma1).exp()*(x1-mu1)
        sigma2 = (1-rho**2)**0.5 * sigma2
        x2 = torch.normal(mu2[:,idx], sigma2[:,idx])#$\Delta
        p_bernoulli = nn.functional.sigmoid(z0_logits.view(1))
        eos = torch.bernoulli(p_bernoulli)
        return torch.cat([x1,x2,eos]).view(1,1,3)

    def generate(self, x0, hidden=None, n=100):
        res = []
        sample = x0
        for i in range(n):
            mu1, mu2, log_sigma1, log_sigma2, rho, pi_logits, z0_logits, hidden = \
            self.forward(sample, hidden)
            sample = self.draw_one_sample(mu1, mu2, log_sigma1, log_sigma2, rho, pi_logits, z0_logits)
            res.append(sample.data.tolist()[0][0])
        return res


network = GaussianHandWriting(dropout=0.2)
optimizer = optim.Adam(network.parameters(), lr=0.005)
max_epoch = 2000
max_norm = 10 # for gradient clipping
info_freq = 10
save_freq = 100


batch = batch_generator()
loss_log = []
hidden = None
for epoch in range(max_epoch):
    x = next(batch)
    optimizer.zero_grad()
    loss, hidden = network.calculate_loss(x[:-1], x[1:], hidden)
    loss.backward()
    torch.nn.utils.clip_grad_norm(network.parameters(), max_norm)
    optimizer.step()
    hidden.detach_()
    loss_log.append(loss.data.tolist())
    if epoch%info_freq == 0:
        print('epoch {}, loss {}'.format(epoch, loss.item()))
    if epoch%save_freq ==0:
        # save model
        torch.save(network.state_dict(),'net_epoch_{:06d}.pt'.format(epoch))
        # check performance
        x0 = Variable(torch.Tensor([0,0,1]).view(1,1,3))
        data = network.generate(x0)
        plot_points(data)
        plt.title('epoch {}'.format(epoch))
        plt.show()










