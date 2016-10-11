from random import choice
from numpy import array, dot, random, loadtxt
from math import exp
#import csv

training = loadtxt("mnist_train.csv", delimiter=",", dtype="int")
#print(a[0,0:1]);
#print(a[0,1:-1]);
#print(a[1]);
#with open('mnist_train.csv', newline='') as csvfile:
    #reader = csv.reader(csvfile, delimiter=',')
    #for row in reader:
    #	if i == 0:
    #		print(row[0])
    #		i+=i+1

unit_step = lambda x: 0 if x < 0 else 1

normalization = lambda x: -1 if x != 0 else 1

def sigmoid(output):
	try:
		res = 2/ (1 + exp(-2 * output))
	except OverflowError:
		res = 0.0
	return res

training_data = [
	(array([0,0,1]), 0),
	(array([0,1,1]), 1),
	(array([1,0,1]), 1),
	(array([1,1,1]), 1),
]

w = random.rand(783)
errors = []
eta = 0.2
n = 100

for i in range(n):
	choose = choice(training)
	x = choose[1:-1]
	expected = normalization(choose[0:1])
	#print(x)
	#print(expected)
	#x, expected = choice(training_data)
	result = dot(w, x)
	sig = sigmoid(result)
	error = expected - sig
	errors.append(error)
	w += eta * error * x

print(errors)

#for x, _ in training:
#	result = dot(x, w)
#	print("{}: {} -> {}".format(x[:2], result, unit_step(result)))