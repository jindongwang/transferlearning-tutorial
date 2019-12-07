# TCA方法的Matlab实现

## 数据获取

由于我们要测试非深度方法，因此，选择SURF特征文件作为算法的输入。SURF特征文件可以从[网络上](https://pan.baidu.com/s/1bp4g7Av)下载。下载到的文件主要包含4个.mat文件：Caltech.mat, amazon.mat, webcam.mat, dslr.mat。它们恰巧对应4个不同的领域。彼此之间两两一组，就是一个迁移学习任务。每个数据文件包含两个部分：fts为800维的特征，labels为对应的标注。在测试中，我们选择由Caltech.mat作为源域，由amazon.mat作为目标域。Office+Caltech10数据集的介绍可以在本手册的附录部分找到。

我们对数据进行加载并做简单的归一化，将最后的数据存入$$X_s,Y_s,X_t,Y_t$$这四个变量中。这四个变量分别对应源域的特征和标注、以及目标域的特征和标注。代码如下：

```
load('Caltech.mat');     % source domain
fts = fts ./ repmat(sum(fts,2),1,size(fts,2)); 
Xs = zscore(fts,1);    clear fts
Ys = labels;           clear labels

load('amazon.mat');    % target domain
fts = fts ./ repmat(sum(fts,2),1,size(fts,2)); 
Xt = zscore(fts,1);     clear fts
Yt = labels;            clear labels
```

## 算法精炼

TCA主要进行边缘分布自适应。通过整理化简，TCA最终的求解目标是：
$$
\left(\mathbf{X} \mathbf{M} \mathbf{X}^\top + \lambda \mathbf{I}\right) \mathbf{A} =\mathbf{X} \mathbf{H} \mathbf{X}^\top \mathbf{A} \Phi 
$$

上述表达式可以通过Matlab自带的`eigs()`函数直接求解。$$\mathbf{A}$$就是我们要求解的变换矩阵。下面我们需要明确各个变量所指代的含义：

- $$\mathbf{X}$$: 由源域和目标域数据共同构成的数据矩阵
- $$C$$: 总的类别个数。在我们的数据集中，$$C=10$$
- $$\mathbf{M}_c$$: MMD矩阵。当$$c=0$$时为全MMD矩阵；当$$c>1$$时对应为每个类别- 的矩阵。
- $$\mathbf{I}$$：单位矩阵
- $$\lambda$$：平衡参数，直接给出
- $$\mathbf{H}$$: 中心矩阵，直接计算得出
- $$\Phi$$: 拉格朗日因子，不用理会，求解用不到

## 编写代码

我们直接给出精炼后的源码：

```
function [X_src_new,X_tar_new,A] = TCA(X_src,X_tar,options)
% The is the implementation of Transfer Component Analysis.
% Reference: Sinno Pan et al. Domain Adaptation via Transfer Component Analysis. TNN 2011.

% Inputs: 
%%% X_src          :    source feature matrix, ns * n_feature
%%% X_tar          :    target feature matrix, nt * n_feature
%%% options        :    option struct
%%%%% lambda       :    regularization parameter
%%%%% dim          :    dimensionality after adaptation (dim <= n_feature)
%%%%% kernel_tpye  :    kernel name, choose from 'primal' | 'linear' | 'rbf'
%%%%% gamma        :    bandwidth for rbf kernel, can be missed for other kernels

% Outputs: 
%%% X_src_new      :    transformed source feature matrix, ns * dim
%%% X_tar_new      :    transformed target feature matrix, nt * dim
%%% A              :    adaptation matrix, (ns + nt) * (ns + nt)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Set options
lambda = options.lambda;              
dim = options.dim;                    
kernel_type = options.kernel_type;    
gamma = options.gamma;                

%% Calculate
X = [X_src',X_tar'];
X = X*diag(sparse(1./sqrt(sum(X.^2))));
[m,n] = size(X);
ns = size(X_src,1);
nt = size(X_tar,1);
e = [1/ns*ones(ns,1);-1/nt*ones(nt,1)];
M = e * e';
M = M / norm(M,'fro');
H = eye(n)-1/(n)*ones(n,n);
if strcmp(kernel_type,'primal')
[A,~] = eigs(X*M*X'+lambda*eye(m),X*H*X',dim,'SM');
Z = A' * X;
Z = Z * diag(sparse(1./sqrt(sum(Z.^2))));
X_src_new = Z(:,1:ns)';
X_tar_new = Z(:,ns+1:end)';
else
K = TCA_kernel(kernel_type,X,[],gamma);
[A,~] = eigs(K*M*K'+lambda*eye(n),K*H*K',dim,'SM');
Z = A' * K;
Z = Z*diag(sparse(1./sqrt(sum(Z.^2))));
X_src_new = Z(:,1:ns)';
X_tar_new = Z(:,ns+1:end)';
end
end

% With Fast Computation of the RBF kernel matrix
% To speed up the computation, we exploit a decomposition of the Euclidean distance (norm)
%
% Inputs:
%       ker:    'linear','rbf','sam'
%       X:      data matrix (features * samples)
%       gamma:  bandwidth of the RBF/SAM kernel
% Output:
%       K: kernel matrix
%
% Gustavo Camps-Valls
% 2006(c)
% Jordi (jordi@uv.es), 2007
% 2007-11: if/then -> switch, and fixed RBF kernel
% Modified by Mingsheng Long
% 2013(c)
% Mingsheng Long (longmingsheng@gmail.com), 2013
function K = TCA_kernel(ker,X,X2,gamma)

switch ker
case 'linear'

if isempty(X2)
K = X'*X;
else
K = X'*X2;
end

case 'rbf'

n1sq = sum(X.^2,1);
n1 = size(X,2);

if isempty(X2)
D = (ones(n1,1)*n1sq)' + ones(n1,1)*n1sq -2*X'*X;
else
n2sq = sum(X2.^2,1);
n2 = size(X2,2);
D = (ones(n2,1)*n1sq)' + ones(n1,1)*n2sq -2*X'*X2;
end
K = exp(-gamma*D); 

case 'sam'

if isempty(X2)
D = X'*X;
else
D = X'*X2;
end
K = exp(-gamma*acos(D).^2);

otherwise
error(['Unsupported kernel ' ker])
end
end
```

我们将TCA方法包装成函数`TCA}`。注意到TCA是一个无监督迁移方法，不需要接受label作为参数。因此，函数共接受3个输入参数：
  
- $$\mathrm{X_{src}}$$: 源域的特征，大小为$$n_s \times m$$
- $$\mathrm{X_{tar}}$$: 目标域的特征，大小为$$n_t \times m$$
- $$\mathrm{options}$$: 参数结构体，它包含：
    - $$\lambda$$: 平衡参数，可以自由给出
    - $$dim$$: 算法最终选择将数据将到多少维
    - $$kernel type$$: 选择的核类型，可以选择RBF、线性、或无核
    - $$\gamma$$: 如果选择RBF核，那么它的宽度为$$\gamma$ 

函数的输出包含3项：

- $$X_{srcnew}$$: TCA后的源域
- $$X_{tarnew}$$: TCA后的目标域
- $$A$$: 最终的变换矩阵

## 测试算法

我们使用如下的代码对TCA算法进行测试：

```
options.gamma = 2;          % the parameter for kernel
options.kernel_type = 'linear';
options.lambda = 1.0;
options.dim = 20;
[X_src_new,X_tar_new,A] = TCA(Xs,Xt,options);

% Use knn to predict the target label
knn_model = fitcknn(X_src_new,Y_src,'NumNeighbors',1);
Y_tar_pseudo = knn_model.predict(X_tar_new);
acc = length(find(Y_tar_pseudo==Y_tar))/length(Y_tar); 
fprintf('Acc=%0.4f\n',acc);
```

结果显示如下：

```
	Acc=0.4499
```

至此，Matlab版TCA实现完成。完整代码可以在[Github](https://github.com/jindongwang/transferlearning/tree/master/code/traditional/TCA)上找到。