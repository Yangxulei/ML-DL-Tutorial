[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Yangxulei/ML-DL-Tutorial/blob/master/LICENSE)


机器学习/人工智能/自然语言处理资料教程集合

前面基本是我看过的书籍和教程推荐，在最后面有我的学习路线，避免多走弯路，实现高效系统学习。

目录：
- [基础概念和认知](#1)
- [入门](#2)
- [教程](#3)
- [书籍](#4)
- [博客论坛期刊](#5)
- [文章-论文](#6)
- [其它](#7)
- 我的学习路线和计划
# 基础概念和认知：
[机器学习](http://zh.wikipedia.org/zh/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0) 机器学习是近20多年兴起的一门多领域交叉学科，涉及概率论、统计学、逼近论、凸分析、算法复杂度理论等多门学科。机器学习理论主要是设计和分析一些让计算机可以自动“学习”的算法。机器学习算法是一类从数据中自动分析获得规律，并利用规律对未知数据进行预测的算法。因为学习算法中涉及了大量的统计学理论，机器学习与统计推断学联系尤为密切，也被称为统计学习理论。算法设计方面，机器学习理论关注可以实现的，行之有效的学习算法。下面是ML比较完备的知识技能图

![ML知识技能图](http://upload-images.jianshu.io/upload_images/744392-ff8a7eb9953e7f95.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


 - [机器学习与数据挖掘的区别](https://en.wikipedia.org/wiki/Machine_learning#Machine_learning_and_data_mining)
   -  机器学习关注从训练数据中学到已知属性进行预测是方法
   -  数据挖掘侧重从数据中发现未知属性是个概念
- [深度学习与机器学习区别](https://zh.wikipedia.org/wiki/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0)
- [人工智能只是定义](https://zh.wikipedia.org/wiki/%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD)
- [机器学习、统计分析、数据挖掘、神经网络、人工智能、模式识别之间的关系是什么？](https://www.zhihu.com/question/20747381)

 

# 入门

#### 心路历程，别人的经验让我们少走弯路：
 [机器学习入门者学习指南-研究生型入门者的亲身经历](http://www.guokr.com/post/512037/)


# 书籍

### 数学
- [《数学之美》](https://book.douban.com/subject/10750155/)作者：吴军  讲述数学在NLP语音等方面应用
- [《程序员的数学系列》](https://book.douban.com/subject_search?search_text=程序员的数学&cat=1001)这套共三本含高数，线代，统计。写的很好
- [《线性代数应该这样学》](https://book.douban.com/subject/26886299/) 从本质来讲，让你耳目一新，我就感觉线代白学了  
- [《The Elements of Statistical Learning》](https://book.douban.com/subject/1761179/)后期精进可以可以读读，挺难的
- [《统计学习方法》](https://book.douban.com/subject/10590856/)作者：李航 讲解统计的一些方法。是机器学习的数学基础，另外需要线性代数，概率论，统计学等数学知识 **后面阶段少不了推算公式的**



### ML&&NLP

- [《机器学习》](https://book.douban.com/subject/26708119/)西瓜书 作者：周志华。 本人读过入门级拿西瓜做案例很清晰，没有数学基础还是不要看，特别是程序员，因为周老师层次太高偏学术。
- [《DeepLearning》](https://github.com/exacity/deeplearningbook-chinese)花书，最近中文版要出版了,称为圣书，这个可以慢慢啃，绝对要看的。
- [《机器学习》](https://book.douban.com/subject/1102235/)作者Mitchell。Mitchell是机器学习的鼻祖，第一个提出机器学习概念的人。这本书很薄，很简单。内容很陈旧，但是都是机器学习的经典问题。而且，这本书概念清晰正确(很可贵啊，又简单又正确的书，说明作者功力很强)。

- 《神经网络与机器学习》作者：Simon Haykin。 事实上，现在常见的很多机器学习算法都发端于神经网络，像SVM，深度学习，CNN等等。这本书详细的介绍了神经网络及其相关算法的所有细节。如果想深入了解的话，可以看一下。只想运用的话，也可以随便翻翻算法的介绍。
- 《模式识别与机器学习》马春鹏 有数学基础再搞，不然很难受
- 《集体编程智慧》代码丰富，结合理论搞
- 《机器学习算法原理与编程实战》有理论有代码
- 《机器学习实战》推荐
- 《数据挖掘导论》我学习过程中必不可少


-----------------------
-  NLP[自然语言处理怎么最快入门](https://www.zhihu.com/question/19895141/answer/20084186?utm_medium=social&utm_source=weibo)
- [统计自然语言处理](https://book.douban.com/subject/3076996/)
- [信息检索导论](https://book.douban.com/subject/5252170/)

- 资料已经完全够了，没有必要再贪多了，有积累看论文才是王道

# 视频教程
### 一、基本功
数学基础机器学习必要的数学基础主要包括：
- 线性代数
    [英文无障碍的话推荐MIT的Gilbert Strang](https://www.youtube.com/watch?v=ZK3O402wf1c&list=PL49CF3715CB9EF31D)
    [李宏毅也教过线性代数哦](http://speech.ee.ntu.edu.tw/~tlkagk/courses_LA16.html)
    实现不行看看张宇的考研视频也行，再者还是看看程序员的数学
- 统计基础
    - [Introduction to Statistics: Descriptive Statistics](https://www.edx.org/course/introduction-statistics-descriptive-uc-berkeleyx-stat2-1x)
    - [Probabilistic Systems Analysis and Applied Probability](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-041-probabilistic-systems-analysis-and-applied-probability-fall-2010/) | [概率](http://open.163.com/special/Khan/probability.html) ( 可选)
    - [Introduction to Statistics: Inference](https://www.edx.org/course/introduction-statistics-inference-uc-berkeleyx-stat2-3x#.U3nU2vmSxhQ)
    - *以上是国外课程，自己拿大学相关课本看看或者到网易公开课找,推荐《程序员的数学之线性代数》*
- 编程基础
    - Python，R这些从廖雪峰，菜鸟教程还有等等看看文档就行 ，numpy，sk，pandas这些库也要会
    - [Programming for Everybody (Python)](https://www.coursera.org/learn/python)
    - [DataCamp: Learn R with R tutorials and coding challenges(R)](https://www.datacamp.com/)
    - [Introduction to Computer Science:Build a Search Engine & a Social Network](https://https://cn.udacity.com/course/intro-to-computer-science--cs101)

 ### 二、修行(推荐李宏毅然后后吴恩达深度学习课程，然后决定走CV OR NLP 再看CS231N 或者 CS224N)
- [台湾国立大学李宏毅中文教学](https://www.youtube.com/channel/UC2ggjtuuWvxrHHHiaDH1dlQ/playlists)讲课幽默易懂，很多不清楚的看了就懂了，它还有个PPT合集，堪称一天学完深度学习，浅显易懂可以搜搜
- [吴恩达深度学习新课](https://www.coursera.org/specializations/deep-learning)可以搜单个课程这样可以旁听不要钱
  【网易云课堂现在和吴恩达合作了，免费的中文字幕体验更好，作业还在courseras上写吧】
- [2014 Andrew Ng (Stanford)机器学习](https://www.coursera.org/course/ml):英文原版视频 这就是针对自学而设计的，免费还有修课认证。“老师讲的是深入浅出，不用太担心数学方面的东西。而且作业也非常适合入门者，都是设计好的程序框架，有作业指南，根据作业指南填写该完成的部分就行。”(和林轩田课程二选一)
- [2016年 林軒田(国立台湾大学) 機器學習基石 (Machine Learning Foundations) -- 華文的教學講解](https://www.youtube.com/user/hsuantien/playlists)
- [Neural Networks for Machine Learning](https://www.coursera.org/learn/neural-networks) Hinton的大牛课程，可想而知
- [没有公式，没有代码的深度学习入门视频，每集五分钟](https://www.youtube.com/playlist?list=PLjJh1vlSEYgvGod9wWiydumYl8hOXixNu)
- [CS231N Stanford winter 2016FeiFei](https://www.youtube.com/playlist?list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC) 计算机视觉大牛李飞飞的课程
- [CS224N NLP Winter 2017](https://www.youtube.com/playlist?list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6) 搞NLP的可以看看，最新前言课程  [slides](https://web.stanford.edu/class/cs224n/syllabus.html)
- [Udacity 的 ML & DL](https://cn.udacity.com/courses/all)  最近刚看的，DL两个人合作以对话形式，很幽默
- [tornadomeet机器学习笔记](http://www.cnblogs.com/tornadomeet/tag/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/)
- 上面学好了，足够了

### 博客等
WeChat公众号：机器之心，数盟，量子位，新智元，

- [我爱机器学习](http://www.52ml.net/) 

- [MITBBS－ 电脑网络 - 数据科学版](http://www.mitbbs.com/bbsdoc/DataSciences.html) 
- [机器之心](!)
- [果壳 > 机器学习小组](http://www.guokr.com/group/262/)
- [统计之都 » 统计学世界 » 数据挖掘和机器学习
](http://cos.name/cn/forum/22) 
- [北邮人论坛 >> 学术科技 >> 机器学习与数据挖掘
](http://bbs.byr.cn/#!board/ML_DM)


文章-论文

[The Discipline of Machine LearningTom Mitchell](http://www.cs.cmu.edu/~tom/pubs/MachineLearning.pdf) 当年为在CMU建立机器学习系给校长写的东西。
[A Few Useful Things to Know about Machine Learning](http://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf) Pedro Domingos教授的大道理，也许入门时很多概念还不明白，上完公开课后一定要再读一遍。

# 其它  
[矩阵求导](https://zhuanlan.zhihu.com/p/24709748)
知乎找答案

---------------------------------------------------------------------------------------------------------------------------

# 我自己的学习计划

已经大致了解这个领域所用到的知识，根据网上的参考进行1.0阶段的学习，看了前面的知道我对PGM特别感兴趣啊，所以感谢夕小瑶(wx:xixiaoyaoQAQ)按照她的提供的知识结构给自己定个计划。  已经完成的都会带有我在学习过程中见到比较好的资料，或者我自己总的，这样减少大家在学习过程中找资料的成本

## 第一阶段(五月到六月)：基本模型
辅助用视频Ng的courses《machine learning》，台湾国立大学林老师《机器学习基石》、《数据挖掘导论》第4、5章
- [感知机(Perceptron)](http://www.jianshu.com/p/f5c3bce6b7ec)(已完成)
   - [如何理解感知机算法的对偶形式](https://www.zhihu.com/question/26526858)
   - [为什么随机梯度下降方法能够收敛？](https://www.zhihu.com/question/27012077/answer/122359602)
- [线性回归模型(Linear regression model) ](http://www.jianshu.com/p/631889a1d1e3) (已完成)
- [逻辑回归模型(Logistic regression model) ](http://www.jianshu.com/p/4db93e9f38af)（已完成）
- [浅层神经网络(Neural Network) ](http://www.cnblogs.com/subconscious/p/5058741.html)(已完成)
- [支持向量机(Support Vector Machine)](http://www.jianshu.com/p/8c5131f933b5)(已完成)
- [交叉验证(Cross Validation)[[1]]](https://zhuanlan.zhihu.com/p/23798202?refer=rdatamining) [[2]](https://zhuanlan.zhihu.com/p/24825503) [[3]](https://www.zhihu.com/question/39259296) [[4]](http://blog.sina.com.cn/s/blog_67f37e760100yes5.html)(已完成)
- [聚类[1]](https://zhuanlan.zhihu.com/p/21387568) (已完成)
  - [K-Means Model [1]](https://zhuanlan.zhihu.com/p/20432322) [[2]](https://zhuanlan.zhihu.com/p/20445085)
- [内积与映射，线性相关/无关，特征值/特征向量、特征向量、特征分解、矩阵的迹、奇异值分解(SVD)、谱定理](http://www.jianshu.com/p/dcaaf3b58c06)

## 第二阶段（七月~）：最优化(已完成，主要是靠知乎和《统计学习方法》，还得继续深入理解)
复习《概率论与数理统计》、理解《Deep Learning》中的4.3节和4.4节，《Numerical Optimazation》、《最优化理论与方法》袁亚湘，孙文瑜、《统计学习方法》、《数据 挖掘导论》、《机器学习实战》、《智能优化方法》
- 一阶无约束优化算法
  - 梯度下降法（步长的确定方法、线搜索法，信赖域法）
- 二阶无约束优化算法
  - 牛顿法
  - 共轭梯度法
  - 拟牛顿法
- 约束优化算法
  - 线性规划(概念与应用、单纯形法、内点法）
  - 二次规划(概念与应用、对偶法、积极集法）
  - 拉格朗日乘子法的简单认识
- 感知机模型
- K近邻模型
- 朴素贝叶斯模型
- 决策树模型
- 支持向量机模型
- 集成分类器
  - Bagging
  - Boosting
  - Random Forest
- 遗传算法
- 模拟退火
- 禁忌搜索算法
- 蚁群算法
- 粒子群优化算法
- [LDA/PCA](http://www.cnblogs.com/LeftNotEasy/archive/2011/01/08/lda-and-pca-machine-learning.html)
- [SVD](http://www.cnblogs.com/LeftNotEasy/archive/2011/01/19/svd-and-applications.html)

## 第三阶段：模式识别与深度学习
- 贝叶斯决策（《模式分类》）
- 参数估计
- 非参数方法
- 线性判别函数
- 浅层神经网络
    - delta方法
    - BP算法及其优化
    - RBF网络
- 深度神经网络（DL中文版书籍）
   - Hopfield网络
   - 玻尔兹曼机
   - RBM
   - DBN
   - DBM
   - CNN
   - Autoencoder
   - RNN
   - LSTM
   (CNN~LSTM)李宏毅教授的课不要太好了。一听就懂
- 聚类
  - 高斯混合密度
  - K-means
  - 层次聚类
- 决策树与随机森林
- 特征提取与特征选择

## 第四阶段：

应该是各种框架和工程，比赛吧，哈哈，到这个阶段就有自己的方向，现在自己也不知道干啥，哈哈哈




## 贯穿始终：
编程是一定要的，推荐先用Python把常用算法实现一遍，然后把NG深度学习课程作业敲一遍，自己写神经网络，你就会明了，一定要code，code，code。后面用tf+keras等都可以。

知识点：(每一周深入学习一种网络)
《统计学习方法》、《Deep Learning》、《模式分类》

- 前馈神经网络
- 自编码器(Auto-Encoder)递归神经网络(Recursive NN) / 循环神经网络(RNN)/ 卷积神经网络(CNN) / 神经张量网络 (NTN)
- 长短时记忆网络(LSTM) / 卷积长短时神经网络(convLSTM) / 张量递归神经网络(MV-RNN)/递归神经张量网络(RNTN)
- 受限波尔兹曼机(RBM) / 玻尔兹曼机

- 概率图模型
 - 有向图模型->贝叶斯网络
 - 无向图模型->马尔科夫随机场->条件随机场




