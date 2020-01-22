
import argparse
from train import Train

# data class import required




def main():
	parser = argparse.ArgumentParser(description='cifar-10 with pytorch')
	parser.add_argument('--lr', default=0.001, type=float)
	parser.add_argument('--epoch', default=200, type=int)
	parser.add_argument('--train_batch_size', default=100, type=int)
	parser.add_argument('--test_batch_size', default=100, type=int)
	args= parser.parse_args()

	train_net = Train(args)
	train_net.run()


if __name__ == '__main__':
	main()