# ML-NLP-AI-
机器学习/人工智能/自然语言处理资料收集
#ML内家心法资料整理 持续更新ing
目录：
- [基础概念和认知](#1)
- [入门](#2)
- [教程](#3)
- [书籍](#4)
- [博客论坛期刊](#5)
- [文章-论文](#6)
- [其它](#7)

#基础概念和认知：
[机器学习](http://zh.wikipedia.org/zh/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0) 机器学习是近20多年兴起的一门多领域交叉学科，涉及概率论、统计学、逼近论、凸分析、算法复杂度理论等多门学科。机器学习理论主要是设计和分析一些让计算机可以自动“学习”的算法。机器学习算法是一类从数据中自动分析获得规律，并利用规律对未知数据进行预测的算法。因为学习算法中涉及了大量的统计学理论，机器学习与统计推断学联系尤为密切，也被称为统计学习理论。算法设计方面，机器学习理论关注可以实现的，行之有效的学习算法。

![ML和其他学科关系的地铁图](http://upload-images.jianshu.io/upload_images/744392-ff8a7eb9953e7f95.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


 - [机器学习与数据挖掘的区别](https://en.wikipedia.org/wiki/Machine_learning#Machine_learning_and_data_mining)
   -  机器学习关注从训练数据中学到已知属性进行预测是方法
   -  数据挖掘侧重从数据中发现未知属性是个概念
- [深度学习与机器学习区别](https://zh.wikipedia.org/wiki/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0)
- [人工智能只是定义](https://zh.wikipedia.org/wiki/%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD)
- [机器学习、统计分析、数据挖掘、神经网络、人工智能、模式识别之间的关系是什么？](https://www.zhihu.com/question/20747381)

 

#入门

#### 心路历程，别人的经验让我们少走弯路：
 [机器学习入门者学习指南-研究生型入门者的亲身经历](http://www.guokr.com/post/512037/)


#书籍

### 科普
- 《数学之美》作者：吴军
- 《 Mathematician's Lament | 数学家的叹息》作者：by Paul Lockhart
- 《 Think Stats: Probability and Statistics for Programmers | 统计思维：程序员数学之概率统计 》 作者：Allen B. Downey
- 《 A History of Mathematics | 数学史 》作者：Carl B. Boyer
- 《 Journeys Through Genius | 天才引导的历程：数学中的伟大定理 》作者：William Dunham
- 《 The Mathematical Experience | 数学经验 》作者 Philip J.Davis、Reuben Hersh
- 《 Proofs from the Book | 数学天书中的证明 》作者：Martin Aigner、Günter M. Ziegler
- 《 Proofs and Refutations | 证明与反驳－数学发现的逻辑 》作者：Imre Lakatos

###深一层次
- 《统计学习方法》作者：李航 讲解统计的一些方法。是机器学习的数学基础，另外需要线性代数，概率论，统计学等数学知识
- 《机器学习》西瓜书 作者：周志华。 本人读过入门级拿西瓜做案例很清晰。

- 《机器学习》作者Mitchell。Mitchell是机器学习的鼻祖，第一个提出机器学习概念的人。这本书很薄，很简单。内容很陈旧，但是都是机器学习的经典问题。而且，这本书概念清晰正确(很可贵啊，又简单又正确的书，说明作者功力很强)。

- 《神经网络与机器学习》作者：Simon Haykin。 事实上，现在常见的很多机器学习算法都发端于神经网络，像SVM，深度学习，CNN等等。这本书详细的介绍了神经网络及其相关算法的所有细节。如果想深入了解的话，可以看一下。只想运用的话，也可以随便翻翻算法的介绍。
- 《人工智能：一种现代的方法》作者：AIMA。基本上学术界的人们都认为机器学习是人工智能学科的下属分支(另一部分人认为是统计学或者数学的分支)，所以，一本人工智能的书也是学习机器学习可以参考的方面。
- 《模式识别与机器学习》不推荐
- 《集体编程智慧》代码丰富但缺少理论
- 《机器学习算法原理与编程实战》有理论有代码
- 《机器学习实战》推荐
- 更新中
#视频教程
###一、基本功
数学基础机器学习必要的数学基础主要包括：
- 多元微积分，线性代数
 - [Calculus: Single Variable](https://www.coursera.org/learn/single-variable-calculus) | [Calculus One](https://www.coursera.org/learn/calculus1) （可选）
 - [Multivariable Calculus](https://ocw.mit.edu/courses/mathematics/18-02sc-multivariable-calculus-fall-2010/)
 - [Linear Algebra](https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/)
 - 以上是国外课程，自己拿大学相关课本看看或者到网易公开课找
- 统计基础
    - [Introduction to Statistics: Descriptive Statistics](https://www.edx.org/course/introduction-statistics-descriptive-uc-berkeleyx-stat2-1x)
    - [Probabilistic Systems Analysis and Applied Probability](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-041-probabilistic-systems-analysis-and-applied-probability-fall-2010/) | [概率](http://open.163.com/special/Khan/probability.html) ( 可选)
    - [Introduction to Statistics: Inference](https://www.edx.org/course/introduction-statistics-inference-uc-berkeleyx-stat2-3x#.U3nU2vmSxhQ)
    - 以上是国外课程，自己拿大学相关课本看看或者到网易公开课找
- 编程基础
    - Python，R这些从菜鸟教程还有等等看看文档就行 
    - [Programming for Everybody (Python)](https://www.coursera.org/learn/python)
    - [DataCamp: Learn R with R tutorials and coding challenges(R)](https://www.datacamp.com/)
    - [Introduction to Computer Science:Build a Search Engine & a Social Network](https://https://cn.udacity.com/course/intro-to-computer-science--cs101)

 ###二、修行

- [2011 Tom Mitchell(CMU)机器学习](http://www.cs.cmu.edu/~tom/10701_sp11/lectures.shtml):英文原版视频与课件PDF 他的《机器学习》在很多课程上被选做教材，有中文版。
- [2014 Andrew Ng (Stanford)机器学习](https://www.coursera.org/course/ml):英文原版视频 这就是针对自学而设计的，免费还有修课认证。“老师讲的是深入浅出，不用太担心数学方面的东西。而且作业也非常适合入门者，都是设计好的程序框架，有作业指南，根据作业指南填写该完成的部分就行。”
- [2008年Andrew Ng CS229 机器学习-网易公开课](http://v.163.com/special/opencourse/machinelearning.html)也有人不推荐
- [2013年Yaser Abu-Mostafa (Caltech) Learning from Data -- 内容更适合进阶](http://work.caltech.edu/lectures.html)
- [Caltech机器学习视频教程库，每个课题一个视频](http://work.caltech.edu/library/)
- [2014年 林軒田(国立台湾大学) 機器學習基石 (Machine Learning Foundations) -- 内容更适合进阶，華文的教學講解] edx或者course上找找
- [中文2012年余凯(百度)张潼(Rutgers) 机器学习公开课 -- 内容更适合进阶](http://wenku.baidu.com/course/view/49e8b8f67c1cfad6195fa705)
博客论坛

- [tornadomeet机器学习笔记](http://www.cnblogs.com/tornadomeet/tag/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/)


*中文:*

- [我爱机器学习](http://www.52ml.net/) 

- [MITBBS－ 电脑网络 - 数据科学版](http://www.mitbbs.com/bbsdoc/DataSciences.html) 
- [机器之心](!)
- [果壳 > 机器学习小组](http://www.guokr.com/group/262/)
- [统计之都 » 统计学世界 » 数据挖掘和机器学习
](http://cos.name/cn/forum/22) 
- [北邮人论坛 >> 学术科技 >> 机器学习与数据挖掘
](http://bbs.byr.cn/#!board/ML_DM)

英文

- [机器学习资源大全
](https://github.com/josephmisiti/awesome-machine-learning)
- [数据挖掘名站](http://www.kdnuggets.com/)
- [数据科学中心网站](http://www.datasciencecentral.com/) 

文章-论文

[The Discipline of Machine LearningTom Mitchell](http://www.cs.cmu.edu/~tom/pubs/MachineLearning.pdf) 当年为在CMU建立机器学习系给校长写的东西。
[A Few Useful Things to Know about Machine Learning](http://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf) Pedro Domingos教授的大道理，也许入门时很多概念还不明白，上完公开课后一定要再读一遍。

#其它  
[矩阵求导](https://zhuanlan.zhihu.com/p/24709748)
