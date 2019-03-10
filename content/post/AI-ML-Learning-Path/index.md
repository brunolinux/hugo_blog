---
date: "2018-12-02"
tags: ["AI"]
title: AI ML Learning Path
---

## 1. Python 入门篇 

- [廖雪峰的 Python 教程](https://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000) 

  推荐看到 正则表达式就可以了，前面如果不懂，可以利用丰富的网上资源查找资料，不用只局限一篇博客

- 书籍: [Learning Python](www.dsf.unica.it/~fiore/LearningPython.pdf) 

  这本书非常值得推荐，感觉写的很详细，但可能对初学者不太友好。

  我在学了 Python 很久后才看到这本书的，感觉很多原先不懂的地方这里都解释得很清楚 

  我的博客中关于 Python 语法部分大多数都是总结自这本书 

- Numpy, Scikit-learn, Matplotlib 学习: [Scipy Lecture Note](https://www.scipy-lectures.org/) (很遗憾缺少 Pandas 库)

- Python 可视化: [Python graph gallery](https://python-graph-gallery.com/)

  注: 我的博客中讲解 seaborn 的部分来自这个网址

注: 

1. 如何寻找有用的 Python 库，不重复造轮子 .

   [awesome-python-cn](https://github.com/jobbole/awesome-python-cn)

2. 这里强烈建议读完前两个，后面关于库的使用，可以在项目中学，后面在 Kaggle 篇会解释

<!--more-->

## 2. Machine learning 入门篇 

- Coursera 的课程 [Machine Learning](https://www.coursera.org/learn/machine-learning/home/welcome) 和 [Deeplearning.ai](https://www.coursera.org/learn/neural-networks-deep-learning/home/welcome) (一共有 5 节课)，注意这两门课在 youtube 都有视频，但没有课后作业。我觉得这门课的精华之处就在课后作业，值得好好做一做。前者用 Matlab/Octave，后者用 Python Jupyter。

- 还有台湾大学林宏毅教授的视频 : 

   [http://speech.ee.ntu.edu.tw/~tlkagk/courses.html](http://speech.ee.ntu.edu.tw/~tlkagk/courses.html)

  (我还没有看，但感觉很不错，我也得抓紧看下)

- [Fast.ai](http://www.fast.ai/) : 对新手入门很好，先跑例程，可以直接看到效果，但没有过多解释背后的数学原理，不然课程怎么叫 fast ai。

- 李沐的 [动手学深度学习](https://zh.gluon.ai/index.html) : MXNet 框架

  **注:** 

  1. 如果对矩阵求导不熟悉，可以参见我的博客 ，请注意 vectorization 的思想非常重要

  2. 如果线性代数忘了，就算没有忘，也请多看看这个网站: [https://ccjou.wordpress.com/](https://ccjou.wordpress.com/) (可能需要梯子)

     或者 MIT 的课程 : [Linear algebra](https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/video-lectures/lecture-1-the-geometry-of-linear-equations/) 以及对应的教材，理解各种矩阵分解和四个子空间

  3. 如果对数学理论感兴趣，可以补充看这几个方面的书籍: 

     凸优化: [Convex optimization](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)

     实变函数和泛函分析: 有点啃不动，有人想学可以一起啊

     数值分析感觉也挺重要的，庆幸我在研究生的时候学过，梯度下降就在这门课讲过，以及范数等

- 如果对传统的机器学习感兴趣，推荐 [Scikit-learn](http://scikit-learn.org/stable/index.html) 库。如果对算法的原理感兴趣，基本上上面这个库的 Tutorial 和 User Guide 都覆盖了，可以看下。

- 入门后的书籍推荐: 

  - Bishop 的 [Pattern Recognition and Machine Learning](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf)  (我正在努力阅读中 :cry: ) 
  - 后面可能还有 ... 

- 后续还有 GAN 对抗生成网络 推荐

## 3. Computer Vision 推荐  

- OpenCV 库  (图像处理库)

  推荐书籍:  [Learning OpenCV3](http://www.bogotobogo.com/cplusplus/files/OReilly%20Learning%20OpenCV.pdf) 

  但还是建议 "在用中学"

- 斯坦福大学课程: 

  - [CS131](http://vision.stanford.edu/teaching/cs131_fall1718/syllabus.html) : introductory course for computer vision
  - CS231a: advanced computer vision
  - CS231n: deep learning and convolution neural networks 

## 4. Pytorch 学习

- [官网 Tutorial](https://pytorch.org/tutorials/)

- 知乎一篇干货: [PyTorch项目代码与资源列表 | 资源下载](https://zhuanlan.zhihu.com/p/28475866)

  但这里大概率会有一个坑，等你们学会 Pytorch 就知道了，这里先不说

## 5. Kaggle 篇 

终于到了 Kaggle 篇，如果前面的 Numpy, Pandas 都没有学过，这里推荐两篇文章: 

- [data science tutorial for beginner](https://www.kaggle.com/kanncaa1/data-sciencetutorial-for-beginners/ )
- [Machine learning tutorial for beginner](https://www.kaggle.com/kanncaa1/machine-learning-tutorial-for-beginners)

在实操几个题目，就可以愉快地玩耍了

- [如何在 Kaggle 首战中进入前 10%](https://dnc1994.com/2016/04/rank-10-percent-in-first-kaggle-competition/) 这篇文章对传统的结构化数据竞赛非常有帮助

## 6. 我的想法

我想在这个暑假创建一个网站，用于展示 神经网络的各种应用，可以和用户交互，比如用户在写字板写入数字，可以自动识别。比如 YoLo 实时物体识别。或者也可以动态展示神经网络的原理。所以在学 Django (一个 Python 的 Web 框架)，慢慢来吧，有兴趣的小伙伴可以加入哦。

