
class myconvNet(nn.Module):

    def __init__(self, image_size=(1,96,96)):
        super(myconvNet, self).__init__()  
        
        self.conv1 = nn.Conv2d(image_size[0], 24, kernel_size=5)
        self.pool1 = torch.nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5)
        self.pool2 = torch.nn.MaxPool2d(2, 2)


        self.conv3 = nn.Conv2d(36, 48, kernel_size=5)
        self.pool3 = torch.nn.MaxPool2d(2, 2)


        self.conv4 = nn.Conv2d(48, 64, kernel_size=5)
        self.pool4 = torch.nn.AvgPool2d(2, 2)
        
        feature_size = self._get_conv_output(image_size)
        
        self.fc1 = nn.Linear(feature_size, 128)    
        self.drop_dense1 = torch.nn.Dropout(0.25)

        self.fc2 = nn.Linear(128, 64)
        self.drop_dense2 = torch.nn.Dropout(0.5)

        self.fc3 = nn.Linear(64, nb_out)

        
    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop_dense1(x)
        x = F.relu(self.fc2(x))
        x = self.drop_dense2(x)
        x = self.fc3(x)
        return x
    
    
    def _forward_features(self, x): 
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)     
        return x
    
    
    def _get_conv_output(self, shape):
        bs = 1
        input = Variable(torch.rand(bs, *shape))
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size
        
model = myconvNet()
if use_gpu:
    myconvNet = myconvNet.cuda()  


