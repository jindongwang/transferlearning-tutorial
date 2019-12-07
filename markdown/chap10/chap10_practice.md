# 第10章 上手实践

以上对迁移学习基本方法的介绍还只是停留在算法的阶段。对于初学者来说，不仅需要掌握基础的算法知识，更重要的是，需要在实验中发现问题。本章的目的是为初学者提供一个迁移学习上手实践的介绍。通过一步步编写代码、下载数据，完成迁移学习任务。在本部分，我们以迁移学习中最为流行的图像分类为实验对象，在流行的Office+Caltech10数据集上完成。

迁移学习方法主要包括：传统的非深度迁移、深度网络的finetune、深度网络自适应、以及深度对抗网络的迁移。教程的目的是抛砖引玉，帮助初学者快速入门。由于网络上已有成型的深度网络的finetune、深度网络自适应、以及深度对抗网络的迁移教程，因此我们不再叙述这些方法，只在这里介绍非深度方法的教程。其他三种方法的地址分别是：

- 深度网络的finetune：
  - [用Pytorch对Alexnet和Resnet进行微调](https://github.com/jindongwang/transferlearning/tree/master/code/deep/finetune_AlexNet_ResNet)
  - [使用PyTorch进行finetune](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- 深度网络的自适应：[DDC/DCORAL方法的Pytorch代码](https://github.com/jindongwang/transferlearning/tree/master/code/deep/DDC_DeepCoral)
- 深度对抗网络迁移：[DANN方法](https://github.com/jindongwang/transferlearning/tree/master/code/deep/DANN(RevGrad))

更多深度迁移方法的代码，请见[这里](https://github.com/jindongwang/transferlearning/tree/master/code/deep)。

在众多的非深度迁移学习方法中，我们选择最经典的迁移方法之一、发表于IEEE TNN 2011的[TCA (Transfer Component Analysis)](https://ieeexplore.ieee.org/abstract/document/5640675/)方法进行实践。为了便于学习，我们同时用Matlab和Python实现了此代码。代码的链接在[这里](https://github.com/jindongwang/transferlearning/tree/master/code/traditional/TCA)。下面我们对代码进行简单讲解。