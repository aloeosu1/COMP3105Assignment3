# COMP 3105 Assignment 3
# Carleton University
# NOTE: This is a sample script to show you how your functions will be called. 
#       You can use this script to visualize your models once you finish your codes. 
#       This script is not meant to be thorough (it does not call all your functions).
#       We will use a different script to test your codes. 
from matplotlib import pyplot as plt

import A3codes as A3codes
from A3helpers import augmentX, plotModel, generateData, plotPoints, plotImgs, synClsExperiments
import pandas


def _plotCls():

	n = 100

	# Generate data
	Xtrain, Ytrain = generateData(n=n, gen_model=1, rand_seed=0)
	Xtrain = augmentX(Xtrain)

	# Learn and plot results
	W = A3codes.minMulDev(Xtrain, Ytrain)
	print(f"Train accuaracy {A3codes.calculateAcc(Ytrain, A3codes.classify(Xtrain, W))}")

	plotModel(Xtrain, Ytrain, W, A3codes.classify)

	# 1d

	train_acc, test_acc = synClsExperiments(A3codes.minMulDev, A3codes.classify, A3codes.calculateAcc)

	print("1d) Training Accuracy Matrix (4 x 2):")
	print(train_acc)

	print("\n1d) Testing Accuracy Matrix (4 x 2):")
	print(test_acc)


	# 2b	testing PCA 
	data = pandas.read_csv('A3train.csv', header=None).values
	U = A3codes.PCA(data, 20)
	plotImgs(U)


	#2d

	train_acc, test_acc = A3codes.synClsExperimentsPCA()

	print("2d) average train acc")
	print(train_acc)
	print("2d) average test acc")
	print(test_acc)

		


	# testing chooseK and repeatKmeans
	Xtrain, Ytrain = generateData(n=100, gen_model=2)
	obj_val_list = A3codes.chooseK(Xtrain)
	
	print("obj_val: ", [float(value) for value in obj_val_list])
	

	return


def _plotKmeans():

	n = 100
	k = 3

	Xtrain, _ = generateData(n, gen_model=2)

	# testing 3a) kmeans
	Y, U, obj_val = A3codes.kmeans(Xtrain, k)
	plotPoints(Xtrain, Y)
	plt.legend()
	plt.show()


	# testing 3b) repeatKmeans
	Y_best, U_best, obj_val_best = A3codes.repeatKmeans(Xtrain, k=3, n_runs=10)
	plotPoints(Xtrain, Y_best)
	plt.legend()
	plt.show()

	return


if __name__ == "__main__":

	_plotCls()
	_plotKmeans()
