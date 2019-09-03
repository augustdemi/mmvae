import argparse
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torchvision import datasets, transforms
from PIL import Image

###############################################################################

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


# -----------------------------------------------------------------------------#

def normal_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.normal_(m.weight, 0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


###############################################################################

class Net(nn.Module):
    """Parametrizes q(z|x).

    @param n_latents: integer
                      number of latent dimensions
    """

    def __init__(self, num_classes=10, channel=1):
        super(Net, self).__init__()
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(channel, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU())


        # self.fc4 = nn.Linear(64*4*4, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, num_classes)

        self.weight_init()

    ####
    def weight_init(self, mode='normal'):

        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init

        for m in self._modules:
            initializer(self._modules[m])

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc4(out))
        out = self.fc5(out)

        return F.log_softmax(out, dim=1)



class NetMnist(nn.Module):
    """Parametrizes q(z|x).

    @param n_latents: integer
                      number of latent dimensions
    """

    def __init__(self, num_classes=10, channel=1):
        super(NetMnist, self).__init__()
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(channel, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU())


        self.fc4 = nn.Linear(64*4*4, 256)
        self.fc5 = nn.Linear(256, num_classes)

        self.weight_init()

    ####
    def weight_init(self, mode='normal'):

        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init

        for m in self._modules:
            initializer(self._modules[m])

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc4(out))
        out = self.fc5(out)

        return F.log_softmax(out, dim=1)



def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))



def main():
    parser = argparse.ArgumentParser()

    # Training settings
    parser.add_argument('--dataset', type=str, default='mnist', help='dataset name')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save_model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--test_only', action='store_true', default=False,
                        help='only test not train')


    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    transformer = transforms.Compose([
                           transforms.Resize(32),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])

    ### choose dataset
    if args.dataset == 'mnist':
        dset_tr = datasets.MNIST('../data/mnist', train=True, download=True,
                       transform=transformer)
        dset_te = datasets.MNIST('../data/mnist', train=False, transform=transformer)
    elif args.dataset == 'fmnist':
        dset_tr = datasets.FashionMNIST(root='../data/fMNIST', train=True, download=True,
                       transform=transformer)
        dset_te = datasets.FashionMNIST(root='../data/fMNIST', train=False, transform=transformer)
    else:
        raise exec('dataset should be mnist or fmnist')

    train_loader = torch.utils.data.DataLoader(
        dset_tr,
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        dset_te,
        batch_size=args.test_batch_size, shuffle=True, **kwargs)




    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    if args.test_only:
        model = torch.load(args.dataset + "_cnn.pt", map_location='cpu')
        test(args, model, device, train_loader)
        test(args, model, device, test_loader)
    else:
        for epoch in range(1, args.epochs +1):
            train(args, model, device, train_loader, optimizer, epoch)
            test(args, model, device, test_loader)
        if (args.save_model):
            torch.save(model, args.dataset + "_cnn2.pt")
            # torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    main()
