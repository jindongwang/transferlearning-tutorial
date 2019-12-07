# 深度网络自适应代码

我们仍然以Pytorch为例，实现深度网络的自适应。具体地说，实现经典的[DDC (Deep Domain Confusion)](https://arxiv.org/abs/1412.3474)方法和与其类似的[DCORAL (Deep CORAL)](https://link.springer.com/chapter/10.1007/978-3-319-49409-8_35)方法。

此网络实现的核心是：如何正确计算DDC中的MMD损失、以及DCORAL中的CORAL损失，并且与神经网络进行集成。此部分对于初学者难免有一些困惑。如何输入源域和目标域、如何进行判断？因此，我们认为此部分应该是深度迁移学习的基础代码，读者应该努力地进行学习和理解。

## 网络结构

首先我们要定义好网络的架构，其应该是来自于已有的网络结构，如Alexnet和Resnet。但不同的是，由于要进行深度迁移适配，因此，输出层要和finetune一样，和目标的类别数相同。其二，由于要进行距离的计算，我们需要加一个叫做bottleneck的层，用来将最高维的特征进行降维，然后进行距离计算。当然，bottleneck层不加尚可。

我们的网络结构如下所示：

```
import torch.nn as nn
import torchvision
from Coral import CORAL
import mmd
import backbone


class Transfer_Net(nn.Module):
def __init__(self, num_class, base_net='resnet50', transfer_loss='mmd', use_bottleneck=True, bottleneck_width=256, width=1024):
super(Transfer_Net, self).__init__()
self.base_network = backbone.network_dict[base_net]()
self.use_bottleneck = use_bottleneck
self.transfer_loss = transfer_loss
bottleneck_list = [nn.Linear(self.base_network.output_num(
), bottleneck_width), nn.BatchNorm1d(bottleneck_width), nn.ReLU(), nn.Dropout(0.5)]
self.bottleneck_layer = nn.Sequential(*bottleneck_list)
classifier_layer_list = [nn.Linear(self.base_network.output_num(), width), nn.ReLU(), nn.Dropout(0.5),
nn.Linear(width, num_class)]
self.classifier_layer = nn.Sequential(*classifier_layer_list)

self.bottleneck_layer[0].weight.data.normal_(0, 0.005)
self.bottleneck_layer[0].bias.data.fill_(0.1)
for i in range(2):
self.classifier_layer[i * 3].weight.data.normal_(0, 0.01)
self.classifier_layer[i * 3].bias.data.fill_(0.0)

def forward(self, source, target):
source = self.base_network(source)
target = self.base_network(target)
source_clf = self.classifier_layer(source)
if self.use_bottleneck:
source = self.bottleneck_layer(source)
target = self.bottleneck_layer(target)
transfer_loss = self.adapt_loss(source, target, self.transfer_loss)
return source_clf, transfer_loss

def predict(self, x):
features = self.base_network(x)
clf = self.classifier_layer(features)
return clf
```

其中Transfer Net是整个网络的模型定义。它接受参数有：


- num class: 目标域类别数
- base net: 主干网络，例如Resnet等，也可以是自己定义的网络结构
- Transfer loss: 迁移的损失，比如MMD和CORAL，也可以是自己定义的损失
- use bottleneck: 是否使用bottleneck
- bottleneck width: bottleneck的宽度
- width: 分类器层的width

## 迁移损失定义

迁移损失是核心。其定义如下：

```
 def adapt_loss(self, X, Y, adapt_loss):
"""Compute adaptation loss, currently we support mmd and coral
Arguments:
X {tensor} -- source matrix
Y {tensor} -- target matrix
adapt_loss {string} -- loss type, 'mmd' or 'coral'. You can add your own loss
Returns:
[tensor] -- adaptation loss tensor
"""
if adapt_loss == 'mmd':
mmd_loss = mmd.MMD_loss()
loss = mmd_loss(X, Y)
elif adapt_loss == 'coral':
loss = CORAL(X, Y)
else:
loss = 0
return loss
```

其中的MMD和CORAL是自己实现的两个loss，MMD对应DDC方法，CORAL对应DCORAL方法。其代码在上述github中可以找到，我们不再赘述。

## 训练

训练时，我们一次输入一个batch的源域和目标域数据。为了方便，我们使用pytorch自带的dataloader。

```
def train(source_loader, target_train_loader, target_test_loader, model, optimizer, CFG):
len_source_loader = len(source_loader)
len_target_loader = len(target_train_loader)
train_loss_clf = utils.AverageMeter()
train_loss_transfer = utils.AverageMeter()
train_loss_total = utils.AverageMeter()
for e in range(CFG['epoch']):
model.train()
iter_source, iter_target = iter(
source_loader), iter(target_train_loader)
n_batch = min(len_source_loader, len_target_loader)
criterion = torch.nn.CrossEntropyLoss()
for i in range(n_batch):
data_source, label_source = iter_source.next()
data_target, _ = iter_target.next()
data_source, label_source = data_source.to(
DEVICE), label_source.to(DEVICE)
data_target = data_target.to(DEVICE)

optimizer.zero_grad()
label_source_pred, transfer_loss = model(data_source, data_target)
clf_loss = criterion(label_source_pred, label_source)
loss = clf_loss + CFG['lambda'] * transfer_loss
loss.backward()
optimizer.step()
train_loss_clf.update(clf_loss.item())
train_loss_transfer.update(transfer_loss.item())
train_loss_total.update(loss.item())
if i % CFG['log_interval'] == 0:
print('Train Epoch: [{}/{} ({:02d}%)], cls_Loss: {:.6f}, transfer_loss: {:.6f}, total_Loss: {:.6f}'.format(
e + 1,
CFG['epoch'],
int(100. * i / n_batch), train_loss_clf.avg, train_loss_transfer.avg, train_loss_total.avg))

# Test
test(model, target_test_loader)
```

完整代码可以在[这里](https://github.com/jindongwang/transferlearning/tree/master/code/deep/DDC_DeepCoral)找到。