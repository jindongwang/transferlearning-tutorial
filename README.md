# README

## 《迁移学习简明手册》

[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT) [![GitHub release](https://img.shields.io/badge/Github-V1.0-519dd9.svg)](https://github.com/jindongwang/transferlearning-tutorial) [![GitHub commits](https://img.shields.io/badge/commits-1-519dd9.svg)](https://github.com/jindongwang/transferlearning-tutorial/issues) ![](https://img.shields.io/badge/language-Tex-orange.svg)

这是《迁移学习简明手册》的LaTex源码。欢迎有兴趣的学者一起来贡献维护。

## New: I've put the LaTex src code on Overleaf, where editing and previewing will be much easier!

## New: I also put the book on GitBook, where everyone can easily see it and modify it.

[**Overleaf address**](https://www.overleaf.com/read/gytccgktstsy)

[**GitBook address**](https://jindongwang.gitbook.io/transfer-learning-tutorial/)

## I'm considering a big update to this tutorial later this year. If you are interested, please feel free to contact me!

### 下载

* V1.1版本: [地址1](http://jd92.wang/assets/files/transfer_learning_tutorial_wjd.pdf) [地址2](https://github.com/jindongwang/transferlearning-tutorial/releases)
* [手册网站与勘误表](http://t.cn/RmasEFe)

### 意见与建议

对于不足和错误之处，以及新的意见，欢迎到[这里](https://github.com/jindongwang/transferlearning-tutorial/issues/6)留言！

#### 引用

可以按如下方式进行引用：

Jindong Wang et al. Transfer Learning Tutorial. 2018.

王晋东等. 迁移学习简明手册. 2018.

**BibTeX**

```text
@misc{WangTLTutorial2018,
    Author = {Jindon Wang et al.},
    Title = {Transfer Learning Tutorial},
    Url = {https://github.com/jindongwang/transferlearning-tutorial},
    Year = {2018},
}

@misc{WangTLTutorial2018cn,
    Author = {王晋东等},
    Title = {迁移学习简明手册},
    Url = {https://github.com/jindongwang/transferlearning-tutorial},
    Year = {2018},
}
```

### 参与贡献方式

以下部分为参与贡献的详细说明。

#### 在线编译 \(推荐\)

直接通过pull request的方式在`markdown`文件夹中修改。修改通过后，GitBook会自动更新。

#### 本地编译方式

* 在任何装有较新版TexLive的电脑上，首先选择`xelatex`引擎进行第一次编译
* 再选择`BibTeX`编译一次生成参考文献
* 最后选择`xelatex`引擎进行第三次编译即可生成带书签的PDF文档

#### 主要文件介绍

以下是本手册的主要文件与其内容介绍：

| 章节 | 名称 | 文件名 | 内容 | 状态 |
| :---: | :---: | :---: | :---: | :---: |
| 主文件 | .. | main.tex | 题目、摘要、推荐语、目录、文件组织 | V1.0 |
| 写在前面等 | .. | prefix.tex | 写在前面、致谢、说明 | V1.0 |
| 第1章 | 迁移学习基本概念 | introduction.tex | 迁移学习基本介绍 | V1.0 |
| 第2章 | 迁移学习的研究领域 | research\_area.tex | 研究领域 | V1.0 |
| 第3章 | 迁移学习的应用 | application.tex | 应用 | V1.0 |
| 第4章 | 基础知识 | basic.tex | 基础知识 | V1.0 |
| 第5章 | 迁移学习的基本方法 | method.tex | 四类基本方法 | V1.0 |
| 第6章 | 第一类方法：数据分布自适应 | distributionadapt.tex | 数据分布自适应 | V1.0 |
| 第7章 | 第二类方法：特征选择 | featureselect.tex | 特征选择 | V1.0 |
| 第8章 | 第三类方法：子空间学习 | subspacelearn.tex | 子空间学习法 | V1.0 |
| 第9章 | 深度迁移学习 | deep.tex | 深度和对抗迁移方法 | V1.0 |
| 第10章 | 上手实践 | practice.tex | 实践教程 | V1.0 |
| 第11章 | 迁移学习前沿 | future.tex | 展望 | V1.0 |
| 第12章 | 总结语 | conclusion | 总结 | V1.0 |
| 第13章 | 附录 | appendix.tex | 附录 | V1.0 |

所有的源码均在`src`目录下。其中，除去主文件`main.tex`外，所有章节都在`chaps/`文件夹下。

所有的图片都在`figures/`文件夹下。推荐实用eps或pdf格式高清文件。

参考文献采用`bibtex`方式，见`refs.bib`文件。

#### 未来计划

* [ ] 丰富和完善现有的V1.0
* [ ] 单独写一章介绍基于实例的迁移学习方法\(instance-based\)，以及相关的instance selection method，如比较经典的tradaboost等
* [ ] 深度和对抗迁移学习方法分成两章，再结合有关文献进行补充
* [ ] 上手实践部分增加对深度方法的说明
* [ ] ……

#### 参与方式

欢迎有兴趣的学者一起加入，让手册更完善！现阶段有2个branch：master用于开发和完善，V1.0是稳定的1.0版本。后续可根据进度增加更多的branch。

具体参与方式：

* 在[这个issue](https://github.com/jindongwang/transferlearning-tutorial/issues/1)下留言你的Github账号和邮箱，我将你添加到协作者中
* 直接fork，然后将你的修改提交pull request
* 如果不熟悉git，可直接下载本目录，然后将你修改的部分发给我\(jindongwang@outlook.com\)
* 有任何问题，均可以提交issue

贡献之后：

* 在下面的贡献者信息中加入自己的信息。
* 如果是对错误的更正，在`web/transfer_tutorial.html`中的"勘误表"部分加入勘误信息。

#### 如何提交 Pull Request

**准备工作**

1. 在原始代码库上点 Fork ，在自己的账户下开一个分支代码库
2. 将自己的分支克隆到本地
   * `git clone https://github.com/(YOUR_GIT_NAME)/transferlearning-tutorial.git`
3. 将本机自己的 fork 的代码库和 GitHub 上原始作者的代码库 ，即上游（ upstream ）连接起来
   * `git remote add upstream https://github.com/jindongwang/transferlearning-tutorial.git`

**提交代码**

1. 每次修改之前，先将自己的本地分支同步到上游分支的最新状态
   * `git pull upstream master`
2. 作出修改后 push 到自己名下的代码库
3. 在 GitHub 网页端自己的账户下看到最新修改后点击 New pull request 即可

#### 贡献者信息

* [@jindongwang](https://github.com/jindongwang) 王晋东，中国科学院计算技术研究所
* [@Godblesswz](https://github.com/Godblesswz) 万震，重庆大学

