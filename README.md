# 《迁移学习简明手册》

这是《迁移学习简明手册》的LaTex源码。主要目的是方便有兴趣的学者一起来贡献维护。

### 下载

下载地址：[V1.0版本](http://jd92.wang/assets/files/transfer_learning_tutorial_wjd.pdf)

### 编译方式

在任何装有较新版TexLive的电脑上，使用`xelatex`方式进行编译。

### 主要文件介绍

以下是本手册的主要文件与其内容介绍：

| 章节 | 名称 | 文件名 | 内容 | 状态 |
|:----------:|:--------------------------:|:---------------------:|:----------------------------------:|:----:|
| 主文件 | .. | main.tex | 题目、摘要、推荐语、目录、文件组织 | V1.0 |
| 写在前面等 | .. | prefix.tex | 写在前面、致谢、说明 | V1.0 |
| 第1章 | 迁移学习基本概念 | introduction.tex | 迁移学习基本介绍 | V1.0 |
| 第2章 | 迁移学习的研究领域 | research_area.tex | 研究领域 | V1.0 |
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

参考文件采用`bibtex`方式，见`refs.bib`文件。

### 未来计划

- [ ] "目录"这两个字不如为什么天各一方了...
- [ ] 丰富和完善现有的V1.0
- [ ] 单独写一章介绍基于实例的迁移学习方法(instance-based)，以及相关的instance selection method，如比较经典的tradaboost等
- [ ] 深度和对抗迁移学习方法分成两章，再结合有关文献进行补充
- [ ] 上手实践部分增加对深度方法的说明
- [ ] ……

### 参与方式

欢迎有兴趣的学者一起加入，让手册更完善！现阶段有2个branch：master用于开发和完善，V1.0是稳定的1.0版本。后续可根据进度增加更多的branch。

参与方式：

- 在[这个issue](https://github.com/jindongwang/transferlearning-tutorial/issues/1)下留言你的Github账号和邮箱，我将你添加到协作者中
- 直接fork，然后将你的修改提交pull request
- 如果不熟悉git，可直接下载本目录，然后将你修改的部分发给我(jindongwang@outlook.com)

然后在下面的贡献者信息中加入自己的信息。

### 贡献者信息

- [@jindongwang](https://github.com/jindongwang) 王晋东，中国科学院计算技术研究所 
