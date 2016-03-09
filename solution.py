import csv
from random import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
class person:

	def __init__(self, pclass, sex, age, sibsp, parch, fare, survived):
		self.pclass = pclass
		self.sex = sex
		self.age = age
		self.sibsp = sibsp
		self.parch = parch
		self.fare = fare
		self.survived  = survived
	

def parseRow(row):
	pclass = int(row[2])
	sex = 0 if row[4] == 'male' else 1
	age = float(row[5]) if len(row[5]) !=0 else -1
	sibsp = int(row[6])
	parch = int(row[7])
	fare = float(row[9]) if len(row[9]) != 0 else -1
	survived = int(row[1]) 


	return person(pclass, sex, age, sibsp, parch, fare, survived) 


def parseTrainer():
	persons = []
	fl = open("train.csv")
	parsedcsv = csv.reader(fl, delimiter = ',', quotechar = '"') 
	idx = -1
	for data in parsedcsv:
		idx += 1
		if idx == 0:
			continue
		persons.append(parseRow(data))

	return persons


def stats(Yp, Ye):
	return 1.0*sum([1 if v==w else 0 for (v,w) in zip(Yp,Ye)])/len(Ye)


def preprocessKFold():
	persons = parseTrainer()
	k = len(persons)/5
	X = [[v.pclass, v.sex, v.age, v.sibsp, v.parch, v.fare] for v in persons]
	Y = [v.survived for v in persons]
	Z = zip(X,Y)
	shuffle(Z)
	X,Y = zip(*Z)
	Xe = X[-k:]
	Ye = Y[-k:]
	X = X[:-k]
	Y = Y[:-k]
	return X,Y,Xe,Ye


def parseTestRow(row):
	pclass = int(row[1])
	sex = 0 if row[3] == 'male' else 1
	age = float(row[4]) if len(row[4]) !=0 else -1
	sibsp = int(row[5])
	parch = int(row[6])
	fare = float(row[8]) if len(row[8]) != 0 else -1
	survived = -1

	return person(pclass, sex, age, sibsp, parch, fare, survived) 



def parseTester():
	persons = []
	fl = open("test.csv")
	parsedcsv = csv.reader(fl, delimiter = ',', quotechar = '"') 
	idx = -1
	for data in parsedcsv:
		idx += 1
		if idx == 0:
			continue
		persons.append(parseTestRow(data))

	fl.close()
	return persons




def preprocess():
	persons = parseTrainer()
	X = [[v.pclass, v.sex, v.age, v.sibsp, v.parch, v.fare] for v in persons]
	Y = [v.survived for v in persons]
	persons = parseTester()
	Xe = [[v.pclass, v.sex, v.age, v.sibsp, v.parch, v.fare] for v in persons]
	return X,Y,Xe

X,Y,Xe = preprocess()

agein = [v[2] for v in X if v[2] != -1]
farein = [v[5] for v in X if v[5] != -1]
aget = [v[2] for v in Xe if v[2] != -1]
faret = [v[5] for v in Xe if v[5] != -1]
ageinm = sorted(agein)[len(agein)/2]
fareinm = sorted(farein)[len(farein)/2]
agetm = sorted(aget)[len(aget)/2]
faretm = sorted(faret)[len(faret)/2]

for v in X:
	if v[2] == -1:
		v[2] = ageinm
	if v[5] == -1:
		v[5] = fareinm
for v in Xe:
	if v[2] == -1:
		v[2] = agetm
	if v[5] == -1:
		v[5] = faretm


dt = open('test.csv').read().split('\n')[1:-1]
ids = []
for d in dt:
	ids.append(int(d.split(',')[0]))
'''clf = KNeighborsClassifier(n_neighbors=13).fit(X,Y)
Yp = clf.predict(Xe)
print Yp

print stats(Yp,Ye)
clf = svm.SVC()
clf.fit(X,Y)
Yp = clf.predict(Xe)
print stats(Yp,Ye)
clf = svm.LinearSVC()
clf.fit(X,Y)
Yp = clf.predict(Xe)
print stats(Yp,Ye)
clf = tree.DecisionTreeClassifier()
clf.fit(X,Y)
Yp = clf.predict(Xe)
print stats(Yp,Ye)
'''
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X,Y)
Yp = clf.predict(Xe)
pred_file = open("RandomForest100median.csv", "wb")
pred_file_obj = csv.writer(pred_file)
pred_file_obj.writerow(["PassengerId","Survived"])

pred_file_obj.writerows(zip(ids,Yp))












