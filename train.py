import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms as transforms
from model.lenet import Lenet
from model.AlexNet import AlexNet

class Train(object):
	def __init__(self, config):
		self.model = None
		self.lr = config.lr
		self.epoch = config.epoch
		self.train_batch_size = config.train_batch_size
		self.test_batch_size = config.test_batch_size

		self.cuda =  torch.cuda.is_available()
		self.device = None
		self.criterion = None
		self.optimizer = None
		self.schedular = None

		self.train_loader = None
		self.test_loader = None

	def load_data(self):
		train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
		test_transform = transforms.Compose([transforms.ToTensor()])
		train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform = train_transform)
		test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
		self.train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=self.train_batch_size, shuffle=True)
		self.test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=self.test_batch_size, shuffle=False)

	def load_model(self):
		if self.cuda:
			self.device = torch.device('cuda')
			print('Model loaded on : ', self.device)
		else:
			self.device = torch.device('cpu')
		self.model = Lenet().to(self.device)
		# self.model = AlexNet().to(self.device)
		self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
		self.schedular = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[75,150],gamma=0.5)
		self.criterion = nn.CrossEntropyLoss().to(self.device)

	def train(self):
		print('*** Training ***')
		self.model.train()
		train_loss= 0
		train_correct = 0
		total = 0

		for batch, (data, target) in enumerate(self.train_loader):
			data, target = data.to(self.device), target.to(self.device)
			self.optimizer.zero_grad()
			output = self.model(data)
			loss = self.criterion(output, target)
			loss.backward()
			self.optimizer.step()
			train_loss+= loss.item()
			prediction = torch.max(output, 1)
			total+=target.size(0)

		return train_loss

	def test(self):
		print("*** Testing ***")
		self.model.eval()
		test_loss = 0
		with torch.no_grad():
			for batch, (data,target) in enumerate(self.test_loader):
				data, target = data.to(self.device), target.to(self.device)
				output = self.model(data)
				loss = self.criterion(output,target)
				test_loss+=loss.item()
			return test_loss

	def save_model(self):
		save_path = 'lenet_model_cifar10.pth'
		torch.save(self.model, save_path)
		print("Checkpoints saved to {}".format(save_path))

	def run(self):
		self.load_data()
		self.load_model()
		for epoch in range(1, self.epoch+1):
			self.schedular.step(epoch)
			print("### Epoch : %d of " %epoch +str(self.epoch))
			train_result = self.train()
			print(train_result)
			test_result = self.test()
			print(test_result)
			if epoch == self.epoch:
				self.save()
