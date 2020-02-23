**course content**

1. lenet 模型介绍
2. lenet 网络搭建
3. 运用lenet进行图像识别-fashion-mnist数据集

**Convolutional Neural Networks**

使用全连接层的局限性：

- 图像在同一列邻近的像素在这个向量中可能相距较远。它们构成的模式可能难以被模型识别。
- 对于大尺寸的输入图像，使用全连接层容易导致模型过大。

使用卷积层的优势：

- 卷积层保留输入形状。
- 卷积层通过滑动窗口将同一卷积核与不同位置的输入重复计算，从而避免参数尺寸过大。

**LeNet 模型**

LeNet分为卷积层块和全连接层块两个部分。下面我们分别介绍这两个模块。

![img](https://github.com/makeittrue/dssdxx-learning-note/blob/master/images/Task05/leNet/01.png)

卷积层块里的基本单位是卷积层后接平均池化层：卷积层用来识别图像里的空间模式，如线条和物体局部，之后的平均池化层则用来降低卷积层对位置的敏感性。

卷积层块由两个这样的基本单位重复堆叠构成。在卷积层块中，每个卷积层都使用5×5的窗口，并在输出上使用sigmoid激活函数。第一个卷积层输出通道数为6，第二个卷积层输出通道数则增加到16。

全连接层块含3个全连接层。它们的输出个数分别是120、84和10，其中10为输出的类别个数。

下面我们通过Sequential类来实现LeNet模型。

In [1]:

    #import 
    import sys 
    sys.path.append("/home/kesci/input") 
    import d2lzh1981 as d2l 
    import torch 
    import torch.nn as nn 
    import torch.optim as optim 
    import time

In [2]:

    #net 
    class Flatten(torch.nn.Module):  #展平操作    
        def forward(self, x):        
            return x.view(x.shape[0], -1) 
    class Reshape(torch.nn.Module): #将图像大小重定型    
        def forward(self, x):        
            return x.view(-1,1,28,28)      #(B x C x H x W)     

    net = torch.nn.Sequential(     #Lelet                                                      
        Reshape(),    
        nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2), #b*1*28*28  =>b*6*28*28    
        nn.Sigmoid(),                                                           
        nn.AvgPool2d(kernel_size=2, stride=2),                              #b*6*28*28  =>b*6*14*14    
        nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),           #b*6*14*14  =>b*16*10*10    
        nn.Sigmoid(),    
        nn.AvgPool2d(kernel_size=2, stride=2),                              #b*16*10*10  => b*16*5*5    
        Flatten(),                                                          #b*16*5*5   => b*400    
        nn.Linear(in_features=16*5*5, out_features=120),    
        nn.Sigmoid(),    
        nn.Linear(120, 84),    
        nn.Sigmoid(),    
        nn.Linear(84, 10) 
    )

接下来我们构造一个高和宽均为28的单通道数据样本，并逐层进行前向计算来查看每个层的输出形状。

In [3]:

    #print 
    X = torch.randn(size=(1,1,28,28), dtype = torch.float32) 
    for layer in net:    
        X = layer(X)    
        print(layer.__class__.__name__,'output shape: \t',X.shape)

Reshape output shape: 	 torch.Size([1, 1, 28, 28]) <br>
Conv2d output shape: 	 torch.Size([1, 6, 28, 28]) <br>
Sigmoid output shape: 	 torch.Size([1, 6, 28, 28]) <br>
AvgPool2d output shape: 	 torch.Size([1, 6, 14, 14]) <br>
Conv2d output shape: 	 torch.Size([1, 16, 10, 10]) <br>
Sigmoid output shape: 	 torch.Size([1, 16, 10, 10]) <br>
AvgPool2d output shape: 	 torch.Size([1, 16, 5, 5]) <br>
Flatten output shape: 	 torch.Size([1, 400]) <br>
Linear output shape: 	 torch.Size([1, 120]) <br>
Sigmoid output shape: 	 torch.Size([1, 120]) <br>
Linear output shape: 	 torch.Size([1, 84]) <br>
Sigmoid output shape: 	 torch.Size([1, 84]) <br>
Linear output shape: 	 torch.Size([1, 10])

可以看到，在卷积层块中输入的高和宽在逐层减小。卷积层由于使用高和宽均为5的卷积核，从而将高和宽分别减小4，而池化层则将高和宽减半，但通道数则从1增加到16。全连接层则逐层减少输出个数，直到变成图像的类别数10。

![img](https://github.com/makeittrue/dssdxx-learning-note/blob/master/images/Task05/leNet/01.png)

**获取数据和训练模型**

下面我们来实现LeNet模型。我们仍然使用Fashion-MNIST作为训练数据集。

In [4]:

# 数据 
batch_size = 256 
train_iter, test_iter = d2l.load_data_fashion_mnist(    
    batch_size=batch_size, root='/home/kesci/input/FashionMNIST2065') 
print(len(train_iter))

235

为了使读者更加形象的看到数据，添加额外的部分来展示数据的图像

In [5]:

    #数据展示 
    import matplotlib.pyplot as plt 
    def show_fashion_mnist(images, labels):    
        d2l.use_svg_display()    
        # 这里的_表示我们忽略（不使用）的变量    
        _, figs = plt.subplots(1, len(images), figsize=(12, 12))    
        for f, img, lbl in zip(figs, images, labels):        
            f.imshow(img.view((28, 28)).numpy())        
            f.set_title(lbl)        
            f.axes.get_xaxis().set_visible(False)        
            f.axes.get_yaxis().set_visible(False)    
        plt.show() 
    for Xdata,ylabel in train_iter:    
        break 
    X, y = [], [] 
    for i in range(10):    
        print(Xdata[i].shape,ylabel[i].numpy())    
        X.append(Xdata[i]) # 将第i个feature加到X中    
        y.append(ylabel[i].numpy()) # 将第i个label加到y中 
    show_fashion_mnist(X, y)

torch.Size([1, 28, 28]) 3 <br>
torch.Size([1, 28, 28]) 8 <br>
torch.Size([1, 28, 28]) 1 <br>
torch.Size([1, 28, 28]) 4 <br>
torch.Size([1, 28, 28]) 0 <br>
torch.Size([1, 28, 28]) 0 <br>
torch.Size([1, 28, 28]) 4 <br>
torch.Size([1, 28, 28]) 9 <br>
torch.Size([1, 28, 28]) 4 <br>
torch.Size([1, 28, 28]) 7

![img](https://cdn.kesci.com/rt_upload/4FE5CE6E20494BFE898E9D8EAAF30C7B/q5ndypu2sq.svg)

因为卷积神经网络计算比多层感知机要复杂，建议使用GPU来加速计算。我们查看看是否可以用GPU，如果成功则使用cuda:0，否则仍然使用cpu。

In [6]:

    # This function has been saved in the d2l package for future use 
    #use GPU 
    def try_gpu():    
        """If GPU is available, return torch.device as cuda:0; else return torch.device as cpu."""    
        if torch.cuda.is_available():        
            device = torch.device('cuda:0')    
        else:        
            device = torch.device('cpu')    
        return device 

    device = try_gpu() 
    device

Out[6]:

    device(type='cpu')

我们实现evaluate_accuracy函数，该函数用于计算模型net在数据集data_iter上的准确率。

In [7]:

    #计算准确率 
    ''' (1). net.train()  启用 BatchNormalization 和 Dropout，将BatchNormalization和Dropout置为True (2). net.eval() 不启用 BatchNormalization 和 Dropout，将BatchNormalization和Dropout置为False '''
    def evaluate_accuracy(data_iter, net,device=torch.device('cpu')):    
        """Evaluate accuracy of a model on the given data set."""    
        acc_sum,n = torch.tensor([0],dtype=torch.float32,device=device),0    
        for X,y in data_iter:        
            # If device is the GPU, copy the data to the GPU.        
            X,y = X.to(device),y.to(device)        
            net.eval()        
            with torch.no_grad():            
                y = y.long()            
                acc_sum += torch.sum((torch.argmax(net(X), dim=1) == y))  #[[0.2 ,0.4 ,0.5 ,0.6 ,0.8] ,[ 0.1,0.2 ,0.4 ,0.3 ,0.1]] => [ 4 , 2 ]            
                n += y.shape[0]    
        return acc_sum.item()/n

我们定义函数train_ch5，用于训练模型。

In [8]:

    #训练函数 
    def train_ch5(net, train_iter, test_iter,criterion, num_epochs, batch_size, device,lr=None):    
        """Train and evaluate a model with CPU or GPU."""    
        print('training on', device)    
        net.to(device)    
        optimizer = optim.SGD(net.parameters(), lr=lr)    
        for epoch in range(num_epochs):        
            train_l_sum = torch.tensor([0.0],dtype=torch.float32,device=device)        
            train_acc_sum = torch.tensor([0.0],dtype=torch.float32,device=device)        
            n, start = 0, time.time()        
            for X, y in train_iter:            
                net.train()                        
                optimizer.zero_grad()            
                X,y = X.to(device),y.to(device)             
                y_hat = net(X)            
                loss = criterion(y_hat, y)            
                loss.backward()            
                optimizer.step()                        
                
                with torch.no_grad():                
                    y = y.long()                
                    train_l_sum += loss.float()                
                    train_acc_sum += (torch.sum((torch.argmax(y_hat, dim=1) == y))).float()                
                    n += y.shape[0]        
                test_acc = evaluate_accuracy(test_iter, net,device)        
                print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, '              
                    'time %.1f sec'              
                    % (epoch + 1, train_l_sum/n, train_acc_sum/n, test_acc,                 
                        time.time() - start))

我们重新将模型参数初始化到对应的设备device(cpu or cuda:0)之上，并使用Xavier随机初始化。损失函数和训练算法则依然使用交叉熵损失函数和小批量随机梯度下降。

In [9]:

        # 训练 
        lr, num_epochs = 0.9, 10 

        def init_weights(m):    
            if type(m) == nn.Linear or type(m) == nn.Conv2d:        
                torch.nn.init.xavier_uniform_(m.weight) 

        net.apply(init_weights) 
        net = net.to(device) 

        criterion = nn.CrossEntropyLoss()   #交叉熵描述了两个概率分布之间的距离，交叉熵越小说明两者之间越接近 
        train_ch5(net, train_iter, test_iter, criterion,num_epochs, batch_size,device, lr)

training on cpu <br>
 epoch 1, loss 0.0091, train acc 0.100, test acc 0.168, time 21.6 sec <br>
 epoch 2, loss 0.0065, train acc 0.355, test acc 0.599, time 21.5 sec <br>
 epoch 3, loss 0.0035, train acc 0.651, test acc 0.665, time 21.8 sec <br>
 epoch 4, loss 0.0028, train acc 0.717, test acc 0.723, time 21.7 sec <br>
 epoch 5, loss 0.0025, train acc 0.746, test acc 0.753, time 21.4 sec <br>
 epoch 6, loss 0.0023, train acc 0.767, test acc 0.754, time 21.5 sec <br>
 epoch 7, loss 0.0022, train acc 0.782, test acc 0.785, time 21.3 sec <br>
 epoch 8, loss 0.0021, train acc 0.798, test acc 0.791, time 21.8 sec <br>
 epoch 9, loss 0.0019, train acc 0.811, test acc 0.790, time 22.0 sec <br>
 epoch 10, loss 0.0019, train acc 0.821, test acc 0.804, time 22.1 sec

In [10]:

    # test 
    for testdata,testlabe in test_iter:    
        testdata,testlabe = testdata.to(device),testlabe.to(device)    
        break 
    print(testdata.shape,testlabe.shape) net.eval() 
    y_pre = net(testdata) 
    print(torch.argmax(y_pre,dim=1)[:10]) 
    print(testlabe[:10])

torch.Size([256, 1, 28, 28]) torch.Size([256]) <br>
tensor([9, 2, 1, 1, 6, 1, 2, 6, 5, 7]) <br>
tensor([9, 2, 1, 1, 6, 1, 4, 6, 5, 7])

**总结：**

卷积神经网络就是含卷积层的网络。 LeNet交替使用卷积层和最大池化层后接全连接层来进行图像分类。

**评论区好的问题**

问：

在训练测试过程中，每次都对optimizer进行清零处理，为啥结果的acc还是呈现递增的趋势

答：

- optimizer = optim.SGD(net.parameters(), lr=lr)，对优化器清零是清除的梯度，防止梯度累加，为下一波梯度计算留空间。而学习的参数并不清零，所以参数越来越优。
- 和前面梯度优化的例子一样，清零不影响网络的记忆性（不能说是连续的，目前的还很难达到连续性智能）

源自：https://www.boyuai.com/elites/course/cZu18YmweLv10OeV/video/5vieguGCIfvhcBQV_PoXu#comment-de9CxsQeUQ0xi98ZhpN-p

问：

1、在构建网络时，卷积核应该怎么设置呢？这块可以讲解下吗？还是说可以通过梯度传播调整

2、卷积的时候图像个数增加是因为引入多个卷积核吗？

3、构建网络时时是否每次卷积完毕必须引入池化层？还是说这个看网络设计者的调节

答：

1.很多同学都在问这块的问题，如何设计神经网络，其实不是这么课程重点需要关注的问题。

​    实际使用时往往不需要你来设计，基本上都是用经典结构，最多是进行一些改造。

​    那么到底如何设计呢？这块一两句话肯定说不清，可以按照发展顺序阅读经典论文，去寻求一些模型设计经验的线索。

​    LeNet -> AlexNet -> VGG -> GoogleNet -> ResNet -> DenseNet 等等，后面就不说了，还有数不尽的论文等待着去学习。。。

2.这个问题提问我没太看明白 

>我觉得这里提问题的人是想问卷积之后图像通道的个数是否等同于卷积核的个数，应该这么说，卷积核的通道数应该等同于图像的通道数而卷积之后的图像的通道数应该与卷积核的个数相等。(作者个人理解)

3.不是每个卷积层后面都要池化，经常是多个卷积层后面接一个池化层。 

​    池化是为了降维，我们最终希望提取到的是一些抽象的有代表性的特征，而不是很多很多感受野非常小的细节特征，例如纹理，颜色等。

​    而且有的网络也会不使用池化层，而是使用步长>1的卷积层来替代pool层完成降维的工作。

源自：https://www.boyuai.com/elites/course/cZu18YmweLv10OeV/video/5vieguGCIfvhcBQV_PoXu#comment-xZZW3YnwdS7oBCM6kq--1

**课后习题**

1.

关于LeNet，以下说法中错误的是：

LeNet主要分为两个部分：卷积层块和全连接层块

√LeNet的绝大多数参数集中在卷积层块部分

LeNet在连接卷积层块和全连接层块时，需要做一次展平操作

LeNet的卷积层块交替使用卷积层和池化层。

答案解释

选项1：正确，参考LeNet模型的结构

选项2：错误，LeNet模型中，90%以上的参数集中在全连接层块

选项3：正确，参考LeNet模型的结构

选项4：正确，参考LeNet模型的结构

2.

关于卷积神经网络，以下说法中错误的是：

√因为全连接层的参数数量比卷积层多，所以全连接层可以更好地提取空间信息

使用形状为2×2，步幅为2的池化层，会将高和宽都减半

卷积神经网络通过使用滑动窗口在输入的不同位置处重复计算，减小参数数量

在通过卷积层或池化层后，输出的高和宽可能减小，为了尽可能保留输入的特征，我们可以在减小高宽的同时增加通道数

答案解释

选项1：错误，参考视频1分钟左右对全连接层局限性的介绍

选项2：正确，参考LeNet中的池化层

选项3：正确，参考视频1分30左右对卷积层优势的介绍

选项4：正确，参考视频3分钟左右的介绍