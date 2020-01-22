import torch.nn as nn

CLASSES = 10

class AlexNet(nn.Module):
	def __init__(self, num_classes=CLASSES):
		super(AlexNet, self).__init__()
		self.conv_layers = nn.Sequential(
			nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2),
			nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2),
			nn.Conv2d(in_channels=192, out_channels=384,kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2),
									    )
		self.fully_connected_layers = nn.Sequential(
			nn.Dropout(),
			nn.Linear(256*2*2, 4096),
			nn.ReLU(inplace=True),
			nn.Dropout(),
			nn.Linear(4096,4096),
			nn.ReLU(inplace=True),
			nn.Linear(4096, num_classes),
												   )
		print('AlexNet layers created')
	def forward(self, x):
		out = self.conv_layers(x)
		out = out.view(out.size(0),-1)
		out = self.fully_connected_layers(out)
		return out

