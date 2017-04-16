import numpy as np
from sklearn.datasets import load_iris

class IrisData:
	def printData(self):
		iris = load_iris()
		X = iris.data
		y = iris.target
		z = iris.target_names
		f = iris.feature_names
		print(X)
		print(y)
		print(f)
					

if __name__ == "__main__":
	obj = IrisData()
	obj.printData()
