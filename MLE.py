import numpy as np
import csv
import math
from scipy.special import comb

class MLE:
    def __init__(self, bit):
        self.state = np.full(bit, -1)
        self.all_state = np.empty((0, bit), dtype=float)
        self.h  = np.full(bit, 1, dtype=float)
        self.Ih = np.full(bit, 0, dtype=float)
        self.J  = np.full(comb(bit,2, exact=True), 1, dtype=float)
        self.W  = np.full((bit, bit), 0, dtype=float)


    def getexvalue(self, fname):
        Ei = np.array([])
        Ek = np.array([])
        hist = np.loadtxt(fname ,delimiter=',')
        hist = np.delete(hist, 0, axis = 1)
        trial = np.sum(hist)
        hist = hist/trial
        print(hist)
        for i in range(bit):
            Ei = np.append(Ei, np.matmul(self.all_state[:,i], hist))
            for j in range(bit -1 -i):
                Ek = np.append(Ek, np.matmul(self.all_state[:,i]*self.all_state[:,i+j+1], hist))
        return Ei, Ek
    
    def getweight(self, JJ):
        count = 0
        for i in range(bit -1):
            for j in range(bit -1 -i):
                self.W[i, j+1+i] = self.W[j+1+i,i] = JJ[count]/2
                count += 1
    
    def getprob(self, hh, WW, mode):
        H = np.array([])
        P = np.array([])
        Eh = np.array([])
        EJ = np.array([])
        for i in range(2 ** bit):
            ss = self.all_state[i]
            H = np.append(H, (-1 * (np.dot(hh, ss) + np.dot(np.matmul(ss, WW), ss))))
            P = np.exp(-H)/np.sum(np.exp(-H))
        for i in range(bit):
            Eh = np.append(Eh, np.matmul(self.all_state[:,i], P))
            for j in range(bit -1 -i):
                EJ = np.append(EJ, np.matmul(self.all_state[:,i]*self.all_state[:,i+j+1], P))
        if (mode==1):
            print(P)
        return Eh, EJ

    def map(self):
        for i in range(bit):
            self.Ih[i] = 14 * np.sqrt(abs(np.round(self.h[i], decimals=1)))
            self.Ih[i] = np.round(self.Ih[i], decimals=1)
            if (self.h[i] < 0):
                self.Ih[i] = -1 * self.Ih[i]
        #print(self.Ih)

    
    def train(self, Trial, epsilon, fname):      
        #get all possible state
        for i in range(2 ** bit):
            self.all_state = np.append(self.all_state, np.array([self.state]), axis=0)
            for j in range(bit):
                if (i%(2 ** j) == ((2 ** j) -1)):
                    self.state[bit - j -1] = -1 * self.state[bit - j -1]
        
        #get expected value of histogram
        E_i, E_k = self.getexvalue(fname)
        
        #train the Ising model 'Trial' times 
        for n in range(Trial):
            self.getweight(self.J)
            E_h, E_J = self.getprob(self.h, self.W, 0)
            L_h = np.array([])
            L_J = np.array([])
            count = 0
            for i in range(bit):
                L_h = np.append(L_h, (E_i[i] - E_h[i])*epsilon)
                self.h[i] = self.h[i] + L_h[i]
                for j in range(bit -1 -i):
                    L_J = np.append(L_J, (E_k[count] - E_J[count])*epsilon)
                    self.J[count] = self.J[count] + L_J[count]
                    count += 1
        print(self.h, self.J)
    

if __name__ == "__main__":
    fname = input("Enter input file name:")
    Bit = input("Enter the number of qubit:")
    bit = int(Bit)
    
    epsilon = 0.1
    Trial = 100000
    
    mle = MLE(bit)
    mle.train(Trial, epsilon, fname)
    