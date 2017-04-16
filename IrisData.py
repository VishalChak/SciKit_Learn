import numpy as np
from sklearn.datasets import load_iris

class IrisData:
	def printData(self):
		iris = load_iris()
		X = iris.data
		y = iris.target
		z = iris.target_names
		f = iris.feature_names		
		print(X.shape)
		print(y.shape)
		print(f)
		print(z[1])

if __name__ == "__main__":
	obj = IrisData()
	obj.printData()
