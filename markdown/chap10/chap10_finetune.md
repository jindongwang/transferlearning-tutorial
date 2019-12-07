# 深度网络的finetune实现

本小节我们用Pytorch实现一个深度网络的finetune。Pytorch是一个较为流行的深度学习工具包，由Facebook进行开发，在[Github](https://github.com/pytorch/pytorch)上也进行了开源。

Finetune指的是训练好的深度网络，拿来在新的目标域上进行微调。因此，我们假定读者具有基本的Pytorch知识，直接给出finetune的代码。完整的代码可以在[这里](https://github.com/jindongwang/transferlearning/tree/master/code/deep/finetune_AlexNet_ResNet)找到。

我们定义一个叫做finetune的函数，它接受输入的一个已有模型，从目标数据中进行微调，输出最好的模型其结果。其代码如下：

```
def finetune(model, dataloaders, optimizer):
since = time.time()
best_acc = 0.0
acc_hist = []
criterion = nn.CrossEntropyLoss()
for epoch in range(1, N_EPOCH + 1):
# lr_schedule(optimizer, epoch)
print('Learning rate: {:.8f}'.format(optimizer.param_groups[0]['lr']))
print('Learning rate: {:.8f}'.format(optimizer.param_groups[-1]['lr']))
for phase in ['src', 'val', 'tar']:
if phase == 'src':
model.train()
else:
model.eval()
total_loss, correct = 0, 0
for inputs, labels in dataloaders[phase]:
inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
optimizer.zero_grad()
with torch.set_grad_enabled(phase == 'src'):
outputs = model(inputs)
loss = criterion(outputs, labels)
preds = torch.max(outputs, 1)[1]
if phase == 'src':
loss.backward()
optimizer.step()
total_loss += loss.item() * inputs.size(0)
correct += torch.sum(preds == labels.data)
epoch_loss = total_loss / len(dataloaders[phase].dataset)
epoch_acc = correct.double() / len(dataloaders[phase].dataset)
acc_hist.append([epoch_loss, epoch_acc])
print('Epoch: [{:02d}/{:02d}]---{}, loss: {:.6f}, acc: {:.4f}'.format(epoch, N_EPOCH, phase, epoch_loss,
epoch_acc))
if phase == 'tar' and epoch_acc > best_acc:
best_acc = epoch_acc
print()
fname = 'finetune_result' + model_name + \
str(LEARNING_RATE) + str(args.source) + \
'-' + str(args.target) + '.csv'
np.savetxt(fname, np.asarray(a=acc_hist, dtype=float), delimiter=',',
fmt='%.4f')
time_pass = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(
time_pass // 60, time_pass % 60))

return model, best_acc, acc_hist
```

其中，model可以是由任意深度网络训练好的模型，如Alexnet、Resnet等。

另外，有很多任务也需要用到深度网络来提取深度特征以便进一步处理。我们也进行了实现，代码在https://github.com/jindongwang/transferlearning/blob/master/code/feature_extractor中。