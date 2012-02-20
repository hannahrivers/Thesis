import numpy as np
import pprint
pp = pprint.PrettyPrinter()

class NN:
    
    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        self.ni = ni + 1
        self.nh = nh
        self.no = no

        # activations for nodes
        self.ai = [1.0]*self.ni
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no
        
        # create weight matrices and initiate with random values
        self.wi = np.random.random((self.ni, self.nh))
        self.wo = np.random.random((self.nh, self.no))

        # last change in weights for momentum   
        self.ci = np.zeros(shape=(self.ni, self.nh))
        self.co = np.zeros(shape=(self.nh, self.no))

    def update(self, inputs):
        # updates the weights and returns output activiations using the non-linear activiation function: the logistic function [0,1)
 
        # make sure inputs is the right size
        if len(inputs) != self.ni-1:
            raise ValueError, 'error: wrong number of inputs'

        # input activations
        for i in range(self.ni-1):
            #self.ai[i] = 1.0/(1.0+np.exp(-inputs[i]))
            self.ai[i] = inputs[i] 

        # hidden activations sum(wi*xi - bias)
        for j in range(self.nh):
            sigma = 0.0
            for i in range(self.ni):
                sigma += self.ai[i] * self.wi[i][j]
            self.ah[j] = 1.0/(1.0+np.exp(-sigma))

        # output activations sum(wi*xi - bias)
        for j in range(self.no):
            sigma = 0.0
            for i in range(self.nh):
                sigma += self.ah[i] * self.wo[i][j]
            self.ao[j] = 1.0/(1.0+np.exp(-sigma))

        return self.ao


    def backPropagate(self, targets, N, M):
        # runs the backpropogation algorithm and returns the error

        # check to make sure targets is the right size
        if len(targets) != self.no:
            raise ValueError, 'error: wrong number of target values'

        # calculate error terms for output
        output_deltas = np.zeros(self.no)
        for i in range(self.no):
            ao = self.ao[i]
            output_deltas[i] = ao*(1-ao)*(targets[i]-ao)

        # calculate error terms for hidden
        hidden_deltas = np.zeros(self.nh)
        for i in range(self.nh):
            sigma = 0.0
            for j in range(self.no):
                sigma += output_deltas[j] * self.wo[i][j]
            hidden_deltas[i] = self.ah[i] * (1-self.ah[i]) * sigma

        # update output weights
        for i in range(self.nh):
            for j in range(self.no):
                change = output_deltas[j] * self.ah[i]
                self.wo[i][j] = self.wo[i][j] + N*change + M*self.co[i][j]
                self.co[i][j] = change

        # update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j] * self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N*change + M*self.ci[i][j]
                self.ci[i][j] = change

        # calculate error
        error = 0.0
        for i in range(self.no):
            delta = targets[i] - self.ao[i]
            error += 0.5 * delta**2

        return error


    def test(self, patterns):
        # this is the classifier step
        
        sus = []
        for p in patterns:
            su = self.update(p[0])
            #print p[1], '->', su
            sus.append(su[0])
        for i in range(len(sus)):
            patterns[i][2].append(sus[i])

    def train(self, patterns, iterations=5000, N=0.5, M=0.9): 
        # this is the training step
        
        # N: learning rate
        # M: momentum factor
        print 'TRAINING'
        for i in xrange(iterations):  # use xrange to save memory in case iterations is very large
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = [p[1][1]]
                self.update(inputs)
                error += self.backPropagate(targets, N, M)
            if i % 100 == 0:
                print 'error:', error


def bodyFat():
    import csv
    reader = csv.reader(open('bodyfatdata.csv','rb'))
    pattern = []
    for row in reader:
        target = float(row[0])/100
        data = row[1:]
        maxd=[1.1089, 81, 363.15, 77.75, 51.2, 136.2, 148.1, 147.7, 87.3, 49.1, 33.9, 45, 34.9, 21.4] # divide each row by max to normalize, this information I got from the spreadsheet instead of using python since it was quicker
        for i in range(len(data)):
            d = data[i]
            data[i] = float(d)/maxd[i]
        pattern.append([data,[target,None],[]])  # row in pattern = [[data], [target value, binary value], [classification values ... ]]

    # bins of 10
    ones = steps(10,pattern)
    tens = steps(20, ones[1])
    twenties = steps(30, tens[1])
    thirties = steps(40, twenties[1])

    print 'BELOW TEN:'
    for i in ones[0]:
        print i[1][0]
    print '\nBETWEEN TEN AND TWENTY:'
    for i in tens[0]:
        print i[1][0]
    print '\nBETWEEN TWENTY AND THIRTY:'
    for i in twenties[0]:
        print i[1][0]
    print '\nBETWEEN THIRTY AND FORTY:'
    for i in thirties[0]:
        print i[1][0]
    print '\nBETWEEN FORTY AND FIFTY:'
    for i in thirties[1]:
        print i[1][0]         

def steps(n,pattern):
    indices0=[]
    indices1=[]

    n = n/100.0

    # assigns binary values
    for row in pattern:
        if row[1][0]<n:
            row[1][1]=0
        else:
            row[1][1]=1           
        
    n = NN(14,3,1)
    n.train(pattern, 1000) # need to up the iterations for increased accuracy
    n.test(pattern)

    # break into two groups (0 and 1)
    for row in pattern:
        if row[2][-1] < .5:
            indices0.append(row)
        else:
            indices1.append(row)
    
    return [indices0,indices1]


def demo():
    # XOR function
    
    pattern =[
        [[0,0], [0]],
        [[0,1], [1]],
        [[1,0], [1]],
        [[1,1], [0]]]

    n = NN(2, 3, 1)
    n.train(pattern, 2000)
    n.test(pattern)

def demo2():
    # if the binary number is even = 1, odd= 0
    
    pattern=[
        [[0,0,0],[1]],
        [[0,0,1],[0]],
        [[0,1,0],[1]],
        [[0,1,1],[0]],
        [[1,0,0],[1]],
        [[1,0,1],[0]],
        [[1,1,0],[1]],
        [[1,1,1],[0]]]

    n=NN(3,3,1)
    n.train(pattern, 2000)
    n.test(pattern)

def demo3():
    # determines if the point (x,y) is best modeled by the equation y=x (0) or y=x^3 (1)
    
    pattern=[]
    
    for i in range(50):
        x = np.random.random()
        e = np.random.randint(0,2)

        if e==0:
            y=x
        else:
            y=x**3

        pattern.append([[x,y],[e]])

    n=NN(2,3,1)
    n.train(pattern, 2000)
    n.test(pattern)


def demo4():
    # determines if the point (x,y) is best modeled by the equation y=**2+b+4*c (0) or y=a**2+b+3*c (1)
    
    pattern=[]
    
    for i in range(50):
        a = np.random.random()
        b = np.random.random()
        c = np.random.random()
        e = np.random.randint(0,2)

        if e==0:
            y=a**2+b+4*c
        else:
            y=a**2+b+3*c

        pattern.append([[a,b,c,y],[e]])

    n=NN(4,3,1)
    n.train(pattern, 2000)
    n.test(pattern)
    

def classifyIris():
    # a classic problem, classifies types of irises
    
    import csv
    reader = csv.reader(open('irisdata.csv','rb'))
    pattern = []
    for row in reader:
        data = row[:-1]
        target = float(row[-1])
        for i in range(len(data)):
            d = data[i]
            data[i] = float(d)
        pattern.append([data,[target]])

    n = NN(4,3,1)
    n.train(pattern, 2000)
    n.test(pattern)
