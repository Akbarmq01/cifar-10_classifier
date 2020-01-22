import torch.nn as nn

cfg = {
		'VGG11':[64,'M', 128,'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
		'VGG13':[64, 64,'M', 128, 128,'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
		'VGG16':[64, 64,'M', 128, 128,'M', 256, 256, 256, 'M', 512, 512,  512, 'M', 512, 512, 512, 'M'],
		'VGG19':[64, 64,'M', 128, 128,'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']

}

class VGG(nn.Module):
	def __init__(self, VGG_model_name):
		super(VGG, self).__init__()
		self.conv_layers = self.vgg_layers(cfg[VGG_model_name])
		self.fully_connected_layers = nn.Sequential(
									  nn.Dropout(),
									  nn.Linear(512,512),
									  nn.ReLU(inplace=True),
									  nn.Dropout(),
									  nn.Linear(512,512),
									  nn.ReLU(inplace=True)
									  nn.Linear(512,10)
									)

	def vgg_layers(self, layer_architecture):
		layers = []
		in_channel = 3
		for x in layer_architecture:
			if x == 'M':
				layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
			else:
				layers += [nn.Conv2d(in_channel, x, kernel_size=3, padding=1),
						   nn.BatchNorm2d(x),
						   nn.ReLU(inplace=True)]
				in_channel = x
		layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
		return nn.Sequential(*layers)
