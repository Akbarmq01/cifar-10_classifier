import torch.nn as nn
import torch.nn.functional as F

CLASSES =10
class Lenet(nn.Module):
	def __init__(self, num_classes = CLASSES):
		super(Lenet, self).__init__()
		self.conv1 = nn.Conv2d(in_channels=3, out_channels=6,  kernel_size=5)
		self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)		
		self.fc1 = nn.Linear(16*5*5, 120)
		self.fc2 = nn.Linear(120,84)
		self.fc3 = nn.Linear(84,num_classes)
	def forward(self, x):
		out = F.relu(self.conv1(x))
		out = F.max_pool2d(out, 2)
		out = F.relu(self.conv2(out))
		out = F.max_pool2d(out, 2)
		out = out.view(out.size(0), -1)
		out = F.relu(self.fc1(out))
		out = F.relu(self.fc2(out))
		out = self.fc3(out)
		return out