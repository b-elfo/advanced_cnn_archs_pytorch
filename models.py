import torch
import torch.nn as nn

###

class LeNet(nn.Module):
    '''
        LeNet (1998) contains five weighted layers, accepts 
        gray images and outputs one of 10 classes.
    '''
    def __init__(self,
                 num_classes: int = 10,
                 device: str = 'cuda',
                 ):
        super().__init__()
        # first conv block
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2, device=device)
        self.tanh1 = nn.Tanh()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        # second conv block
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, device=device)
        self.tanh2 = nn.Tanh()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # third conv block
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1, device=device)
        self.tanh3 = nn.Tanh()
        # flatten tensor for linear use
        self.flat1 = nn.Flatten()
        # first linear layer
        self.fc1   = nn.Linear(120, 84, device=device)
        # second linear layer
        self.fc2   = nn.Linear(84, num_classes, device=device)

    def forward(self, 
                X: torch.Tensor,
                ):
        # first conv block
        X = self.conv1(X)
        X = self.tanh1(X)
        X = self.pool1(X)
        # second conv block
        X = self.conv2(X)
        X = self.tanh2(X)
        X = self.pool2(X)
        # third conv block
        X = self.conv3(X)
        X = self.tanh3(X)
        # flatten tensor for linear use
        X = self.flat1(X)
        # first linear layer
        X = self.fc1(X)
        # second linear layer (output)
        X = self.fc2(X)

        return X

###

class AlexNet(nn.Module):
    '''
        AlexNet (2012) contains seven weighted layers, accepts 
        RGB images and (normally) outputs 1000 classes
    '''
    def __init__(self,
                 num_classes: int = 1000,
                 inp_channels: int = 3,
                 device: str = 'cuda',
                 ):
        super().__init__()
        # first conv block
        self.conv1 = nn.Conv2d(inp_channels, 96, kernel_size=11, stride=4, padding=5, device=device)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn1   = nn.BatchNorm2d(96, device=device)
        # second conv block
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, device=device)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn2   = nn.BatchNorm2d(256, device=device)
        # third conv block
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1, device=device)
        self.relu3 = nn.ReLU()
        self.bn3   = nn.BatchNorm2d(384, device=device)
        # fourth conv block
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1, device=device)
        self.relu4 = nn.ReLU()
        self.bn4   = nn.BatchNorm2d(384, device=device)
        # fifth conv block
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, device=device)
        self.relu5 = nn.ReLU()
        self.bn5   = nn.BatchNorm2d(256, device=device)
        self.pool3 = nn.AdaptiveMaxPool2d(1)
        # flatten tensor for linear layers
        self.flat1 = nn.Flatten()
        # first linear block
        self.fc1   = nn.Linear(256, 4096, device=device)
        self.relu6 = nn.ReLU()
        self.dout1 = nn.Dropout()
        # second linear block
        self.fc2   = nn.Linear(4096, 4096, device=device)
        self.relu7 = nn.ReLU()
        self.dout2 = nn.Dropout()
        # third linear block (output)
        self.fc3   = nn.Linear(4096, num_classes, device=device)
        self.sftmx = nn.Softmax()

    def forward(self, 
                X: torch.Tensor,
                ):
        # first conv block
        X = self.conv1(X)
        X = self.relu1(X)
        X = self.pool1(X)
        X = self.bn1(X)
        # second conv block
        X = self.conv2(X)
        X = self.relu2(X)
        X = self.pool2(X)
        X = self.bn2(X)
        # third conv block
        X = self.conv3(X)
        X = self.relu3(X)
        X = self.bn3(X)
        # fourth conv block
        X = self.conv4(X)
        X = self.relu4(X)
        X = self.bn4(X)
        # fifth conv block
        X = self.conv5(X)
        X = self.relu5(X)
        X = self.bn5(X)
        X = self.pool3(X)
        # flatten tensor for linear layers
        X = self.flat1(X)
        # first linear block
        X = self.fc1(X)
        X = self.relu6(X)
        X = self.dout1(X)
        # second linear block
        X = self.fc2(X)
        X = self.relu7(X)
        X = self.dout2(X)
        # third linear block
        X = self.fc3(X)
        X = self.sftmx(X)
        
        return X

###

class VGGNet(nn.Module):
    '''
        VGGNet (2014) contains 16 weighted layers, accepts RGB
        images and (normally) outputs 1000 classes.
    '''
    def __init__(self,
                 num_classes: int = 1000,
                 inp_channels: int = 3,
                 device: str = 'cuda',
                 ):
        super().__init__()
        # first conv block
        self.conv1 = nn.Conv2d(inp_channels, 64, kernel_size=3, padding=1, device=device)
        self.relu1 = nn.ReLU()
        # second conv block
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, device=device)
        self.relu2 = nn.ReLU()
        # first pooling layer
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        # third conv block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, device=device)
        self.relu3 = nn.ReLU()
        # fourth conv block
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1, device=device)
        self.relu4 = nn.ReLU()
        # second pooling layer
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        # fifth conv block
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1, device=device)
        self.relu5 = nn.ReLU()
        # sixth conv block
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1, device=device)
        self.relu6 = nn.ReLU()
        # seventh conv block
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1, device=device)
        self.relu7 = nn.ReLU()
        # third pooling layer
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        # eighth conv block
        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, padding=1, device=device)
        self.relu8 = nn.ReLU()
        # ninth conv block
        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, padding=1, device=device)
        self.relu9 = nn.ReLU()
        # tenth conv block
        self.conv10= nn.Conv2d(512, 512, kernel_size=3, padding=1, device=device)
        self.relu10= nn.ReLU()
        # fourth pooling layer
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        # eleventh conv block
        self.conv11= nn.Conv2d(512, 512, kernel_size=3, padding=1, device=device)
        self.relu11= nn.ReLU()
        # twelfth conv block
        self.conv12= nn.Conv2d(512, 512, kernel_size=3, padding=1, device=device)
        self.relu12= nn.ReLU()
        # thirteenth conv block
        self.conv13= nn.Conv2d(512, 512, kernel_size=3, padding=1, device=device)
        self.relu13= nn.ReLU()
        # fifth pooling layer (adaptive ~ flattens)
        self.pool5 = nn.AdaptiveMaxPool2d(1)
        self.flat1 = nn.Flatten()
        # first linear block
        self.fcnl1 = nn.Linear(512, 4096, device=device)
        self.relu14= nn.ReLU()
        self.drpo1 = nn.Dropout()
        # second linear block
        self.fcnl2 = nn.Linear(4096, 4096, device=device)
        self.relu15= nn.ReLU()
        self.drpo2 = nn.Dropout()
        # third linear block
        self.fcnl3 = nn.Linear(4096, num_classes, device=device)
        self.sfmx1 = nn.Softmax()

    def forward(self, 
                X: torch.Tensor,
                ):
        # first conv block
        X = self.conv1(X)
        X = self.relu1(X)
        # second conv block
        X = self.conv2(X)
        X = self.relu2(X)
        # first pooling layer
        X = self.pool1(X)
        # third conv block
        X = self.conv3(X)
        X = self.relu3(X)
        # fourth conv block
        X = self.conv4(X)
        X = self.relu4(X)
        # second pooling layer
        X = self.pool2(X)
        # fifth conv block
        X = self.conv5(X)
        X = self.relu5(X)
        # sixth conv block
        X = self.conv6(X)
        X = self.relu6(X)
        # seventh conv block
        X = self.conv7(X)
        X = self.relu7(X)
        # third pooling layer
        X = self.pool3(X)
        # eighth conv block
        X = self.conv8(X)
        X = self.relu8(X)
        # ninth conv block
        X = self.conv9(X)
        X = self.relu9(X)
        # tenth conv block
        X = self.conv10(X)
        X = self.relu10(X)
        # fourth pooling layer
        X = self.pool4(X)
        # eleventh conv block
        X = self.conv11(X)
        X = self.relu11(X)
        # twelfth conv block
        X = self.conv12(X)
        X = self.relu12(X)
        # thirteenth conv block
        X = self.conv13(X)
        X = self.relu13(X)
        # fifth pooling layer
        X = self.pool5(X)
        # flatten layer
        X = self.flat1(X)
        # first linear layer
        X = self.fcnl1(X)
        X = self.relu14(X)
        X = self.drpo1(X)
        # second linear layer
        X = self.fcnl2(X)
        X = self.relu15(X)
        X = self.drpo2(X)
        # third linear layer
        X = self.fcnl3(X)
        X = self.sfmx1(X)

        return X

###

class InceptionModule(nn.Module):
    '''
        The Inception block allows the GoogLeNet to be deeper than 
        previous models (LeNet, AlexNet) while having less parameters.
        
        It contains four parallel convolutional paths, which rely on 
        bottlenecks to reduce the dimensionality.
    '''
    def __init__(self, 
                 input_1x1: int,
                 output_1x1_1: int,
                 input_3x3: int,
                 output_3x3: int,
                 input_5x5: int,
                 output_5x5: int,
                 output_1x1_4: int,
                 device: str = 'cuda',
                 ):
        '''
            :param input_1x1:    input dimension of all four 1x1 convolution
            :param output_1x1_1: output dimension of first 1x1 convolution
            :param input_3x3:    input dimension of 3x3 convolution
            :param output_3x3:   output dimension of 3x3 convolution
            :param input_5x5:    input dimension of 5x5 convolution
            :param output_5x5:   output dimension of 5x5 convolution
            :param output_1x1_4: output dimension for fourth convolution
        '''
        super().__init__()
        # 1x1 Conv Block
        self.conv1x1_1 = nn.Conv2d(input_1x1, output_1x1_1, kernel_size=1, stride=1, device=device)
        self.relu_1    = nn.ReLU()
        # 3x3 Conv Block
        self.conv1x1_2 = nn.Conv2d(input_1x1, input_3x3, kernel_size=1, stride=1, device=device)
        self.relu_2    = nn.ReLU()
        self.conv3x3_1 = nn.Conv2d(input_3x3, output_3x3, kernel_size=3, stride=1, padding=1, device=device) 
        self.relu_3    = nn.ReLU()
        # 5x5 Conv Block
        self.conv1x1_3 = nn.Conv2d(input_1x1, input_5x5, kernel_size=1, stride=1, device=device)
        self.relu_4    = nn.ReLU()
        self.conv5x5_1 = nn.Conv2d(input_5x5, output_5x5, kernel_size=5, stride=1, padding=2, device=device)
        self.relu_5    = nn.ReLU()
        # Pool Block
        self.maxpool_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv1x1_4 = nn.Conv2d(input_1x1, output_1x1_4, kernel_size=1, stride=1, device=device)
        self.relu_6    = nn.ReLU()

    def forward(self, X):
        # 1x1 Conv Block
        X_1 = self.conv1x1_1(X)
        X_1 = self.relu_1(X_1)
        # 3x3 Conv Block
        X_2 = self.conv1x1_2(X)
        X_2 = self.relu_2(X_2)
        X_2 = self.conv3x3_1(X_2)
        X_2 = self.relu_3(X_2)
        # 5x5 Conv Block
        X_3 = self.conv1x1_3(X)
        X_3 = self.relu_4(X_3)
        X_3 = self.conv5x5_1(X_3)
        X_3 = self.relu_5(X_3)
        # Pool Block
        X_4 = self.maxpool_1(X)
        X_4 = self.conv1x1_4(X_4)
        X_4 = self.relu_6(X_4)
        # Concatenate all output tensors in inception block
        X = torch.cat((X_1,X_2,X_3,X_4), dim=1)
        
        return X

class GoogLeNet(nn.Module):
    '''
        GoogLeNet (2014) contains 67 weighted layers (including nine
        Inception blocks), accepts RGB images and (normally) outputs
        ten classes.
    '''
    def __init__(self,
                 num_classes: int = 10,
                 inp_channels: int = 3,
                 device: str = 'cuda',
                 ):
        super().__init__()
        # first conv block
        self.conv7x7_1 = nn.Conv2d(inp_channels, 64, kernel_size=7, stride=2, padding=3, device=device)
        self.relu_1    = nn.ReLU()
        # first pooling layer
        self.maxpool_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.batchnm_1 = nn.BatchNorm2d(64, device=device)
        # second conv block
        self.conv1x1_1 = nn.Conv2d(64, 64, kernel_size=1, stride=1, device=device)
        self.relu_2    = nn.ReLU()
        # third conv block
        self.conv3x3_1 = nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1, device=device)
        self.relu_3    = nn.ReLU()
        # second pooling layer
        self.maxpool_2 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.batchnm_2 = nn.BatchNorm2d(192, device=device)
        # first inception block
        self.inceptn_1 = InceptionModule(192,64,96,128,16,32,32,device=device)
        self.inceptn_2 = InceptionModule(256,128,128,192,32,96,64,device=device)
        # third pooling layer
        self.maxpool_3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # second inception block
        self.inceptn_3 = InceptionModule(480,192,96,208,16,48,64,device=device)
        self.inceptn_4 = InceptionModule(512,160,112,224,24,64,64,device=device)
        self.inceptn_5 = InceptionModule(512,128,128,256,24,64,64,device=device)
        self.inceptn_6 = InceptionModule(512,112,144,288,32,64,64,device=device)
        self.inceptn_7 = InceptionModule(528,256,160,320,32,128,128,device=device)
        # fourth pooling layer
        self.maxpool_4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # third inception block
        self.inceptn_8 = InceptionModule(832,256,160,320,32,128,128,device=device)
        self.inceptn_9 = InceptionModule(832,384,192,384,48,128,128,device=device)
        # fifth pooling layer
        self.avgpool_1 = nn.AdaptiveAvgPool2d(1)
        self.flatten_1 = nn.Flatten()
        self.dropout_1 = nn.Dropout(0.4)
        # first linear layer
        self.fcnlayr_1 = nn.Linear(384+384+128+128, num_classes, device=device) 
        self.softmax_1 = nn.Softmax()
    
    def forward(self, X):
        # first conv block
        X = self.conv7x7_1(X)
        X = self.relu_1(X)
        # first pooling layer
        X = self.maxpool_1(X)
        X = self.batchnm_1(X)
        # second conv block
        X = self.conv1x1_1(X)
        X = self.relu_2(X)
        # third conv block
        X = self.conv3x3_1(X)
        X = self.relu_3(X)
        # second pooling layer
        X = self.maxpool_2(X)
        X = self.batchnm_2(X)
        # first inception block
        X = self.inceptn_1(X)
        X = self.inceptn_2(X)
        # third pooling layer
        X = self.maxpool_3(X)
        # second inception block
        X = self.inceptn_3(X)
        X = self.inceptn_4(X)
        X = self.inceptn_5(X)
        X = self.inceptn_6(X)
        X = self.inceptn_7(X)
        # fourth pooling layer
        X = self.maxpool_4(X)
        # third inception block
        X = self.inceptn_8(X)
        X = self.inceptn_9(X)
        # fifth pooling layer
        X = self.avgpool_1(X)
        # flatten layer
        X = self.flatten_1(X)
        # dropout layer
        X = self.dropout_1(X)
        # first linear layer
        X = self.fcnlayr_1(X)
        X = self.softmax_1(X)
        
        return X

###

class BottleNeckResBlock(nn.Module):
    '''
        The bottleneck residual block contains three sequential
        convolutional layers (cummulatively a bottleneck), as 
        well as a skip connection. The skip connection can be a 
        bottleneck (additional 1x1 conv layer along skip connection 
        to reduce dimension) or not (simple connection). 
    '''
    def __init__(self, 
                 filters: list,
                 reduce: bool = False,
                 kernel: int = 3,
                 stride: int = 1,
                 device: str = 'cuda',
                 ):
        '''
            :param filters: list of kernel sizes for the convolutional layers
            :param reduce: to use bottleneck on skip connection or not
            :param kernel: dimension of the middle convoluitional kernel
            :param stride: stride of the first convolutional layer, as well as
                           the skip connection convolution if it exists
        '''
        super().__init__()
        self.reduce = reduce
        # unpack filters
        f0,f1,f2,f3 = filters
        # bottleneck
        # first conv block (1x1)
        self.conv1 = nn.Conv2d(f0, f1, kernel_size=1, stride=stride, device=device)
        self.btnm1 = nn.BatchNorm2d(f1, device=device)
        self.relu1 = nn.ReLU()
        # second conv block
        self.conv2 = nn.Conv2d(f1, f2, kernel_size=kernel, stride=1, padding=(kernel-1)//2, device=device)
        self.btnm2 = nn.BatchNorm2d(f2, device=device)
        self.relu2 = nn.ReLU()
        # third conv block (1x1)
        self.conv3 = nn.Conv2d(f2, f3, kernel_size=1, stride=1, device=device)
        self.btnm3 = nn.BatchNorm2d(f3, device=device)
        self.relu3 = nn.ReLU()

        if self.reduce:
            # if reducing, we add the skip bottleneck
            self.conv_btlnk = nn.Conv2d(f0, f3, kernel_size=1, stride=stride, device=device)
            self.btnm_btlnk = nn.BatchNorm2d(f3, device=device)


    def forward(self,
                X: torch.Tensor,
                ):
        # temporary hold input tensor for skip connection
        X_skip = X
        # first conv block
        X = self.conv1(X)
        X = self.btnm1(X)
        X = self.relu1(X)
        # second conv block
        X = self.conv2(X)
        X = self.btnm2(X)
        X = self.relu2(X)
        # third conv block
        X = self.conv3(X)
        X = self.btnm3(X)
        
        if self.reduce:
            # skip conv layer
            X_skip = self.conv_btlnk(X_skip)
            X_skip = self.btnm_btlnk(X_skip)
        # add skip connection to bottleneck output
        X += X_skip
        # output activation
        X = self.relu3(X)

        return X

class ResNet50(nn.Module):
    '''
        ResNet50 (2015) contains 50 weighted layers (hence the name),
        accepts RGB images and (normally) outputs a custom number of 
        classes.
    '''
    def __init__(self,
                 num_classes: int,
                 inp_channels: int = 3,
                 device: str = 'cuda',
                 ):
        '''
            :param num_classes: number of output classes
        '''
        super().__init__()
        # first conv block
        self.conv7x7  = nn.Conv2d(inp_channels, 64, kernel_size=7, stride=2, padding=3, device=device)
        self.batchnm  = nn.BatchNorm2d(64, device=device)
        self.relu_1   = nn.ReLU()
        # first pooling layer
        self.maxpool  = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        # first ResNet block
        self.resblk_1  = BottleNeckResBlock([64,64,64,256], reduce=True, device=device)
        self.resblk_2  = BottleNeckResBlock([256,64,64,256], device=device)
        self.resblk_3  = BottleNeckResBlock([256,64,64,256], device=device)
        # second ResNet block
        self.resblk_4  = BottleNeckResBlock([256,128,128,512], reduce=True, stride=2, device=device)
        self.resblk_5  = BottleNeckResBlock([512,128,128,512], device=device)
        self.resblk_6  = BottleNeckResBlock([512,128,128,512], device=device)
        self.resblk_7  = BottleNeckResBlock([512,128,128,512], device=device)
        # third ResNet block
        self.resblk_8  = BottleNeckResBlock([512,256,256,1024], reduce=True, stride=2, device=device)
        self.resblk_9  = BottleNeckResBlock([1024,256,256,1024], device=device)
        self.resblk_10 = BottleNeckResBlock([1024,256,256,1024], device=device)
        self.resblk_11 = BottleNeckResBlock([1024,256,256,1024], device=device)
        self.resblk_12 = BottleNeckResBlock([1024,256,256,1024], device=device)
        self.resblk_13 = BottleNeckResBlock([1024,256,256,1024], device=device)
        # fourth ResNet block
        self.resblk_14 = BottleNeckResBlock([1024,512,512,2048], reduce=True, stride=2, device=device)
        self.resblk_15 = BottleNeckResBlock([2048,512,512,2048], device=device)
        self.resblk_16 = BottleNeckResBlock([2048,512,512,2048], device=device)
        # second pooling layer
        self.avgpool_1 = nn.AdaptiveAvgPool2d(1)
        self.flatten_1 = nn.Flatten()
        # first linear layer (output)
        self.fcn_1     = nn.Linear(2048, num_classes, device=device)
    
    def forward(self,
                X: torch.Tensor,
                ):
        # first conv block
        X = self.conv7x7(X)
        X = self.batchnm(X)
        X = self.relu_1(X)
        # first pooling layer
        X = self.maxpool(X)
        # first ResNet block
        X = self.resblk_1(X)
        X = self.resblk_2(X)
        X = self.resblk_3(X)
        # second ResNet block
        X = self.resblk_4(X)
        X = self.resblk_5(X)
        X = self.resblk_6(X)
        X = self.resblk_7(X)
        # third ResNet block
        X = self.resblk_8(X)
        X = self.resblk_9(X)
        X = self.resblk_10(X)
        X = self.resblk_11(X)
        X = self.resblk_12(X)
        X = self.resblk_13(X)
        # fourth ResNet block
        X = self.resblk_14(X)
        X = self.resblk_15(X)
        X = self.resblk_16(X)
        # second pooling layer
        X = self.avgpool_1(X)
        # flatten layer
        X = self.flatten_1(X)
        # first linear layer (output)
        X = self.fcn_1(X)

        return X