from os import terminal_size
import numpy as np
import random
import time

class Network():
    def __init__(self,layer_sizes,filename = None):
        #Initialisieren des Netzwerks, layer_sizes gibt Anzahl der Neuronen
        #der Layers in einem Array an, Bsp.: [100,20,5] wären 3 Layer, Layer 1
        #hat 100 Neuronen, Layer 2 hat 20 Neuronen, Layer 3 hat 5 Neuronen.
        #filename: name des files, aus dem weights/biases geladen werden sollen
        #wenn filename weggelassen wird: ws/bb werden "frisch" random generiert

        #Biases: für jeden Layer mit n Neuronen jeweils eine (nx1) Matrix, außer
        #für den ersten Layer, für den man keine Biases benötigt
        #Weights: für Bereich zwischen Layer 1 mit n1 Neuronen und Layer 2 mit n2
        #jeweils eine zufällige (n2xn1) Matrix
        #np.random.randn(m,n) erzeugt zufällige (mxn)-Matrix,
        #Werte entsprechen Gaußscher Normalverteilung, Schnitt = 0, Varianz = 1
        self.layer_sizes = layer_sizes
        self.percentage = 0
        if filename:
            self.loadNetwork(filename)
        else:
            self.biases = [np.random.randn(m,1) for m in layer_sizes[1:]]
            self.weights = []
            for n in range(len(layer_sizes)-1):
                self.weights.append(np.random.randn(layer_sizes[n+1],layer_sizes[n]))
        
        #Bestätigung ausgeben
        print("\nCreated neural network, layer sizes:",self.layer_sizes)
        if filename:
            print("Loaded existing network from "+filename+".npy")

    def feed(self,a):
        #Netzwerk erhält in a Liste der Werte für den ersten Layer
        #Dieser 1xn-Vektor wird in nx1-Vektor transponiert
        #Beginnt bei Layer 1, berechnet mithilfe w/b was in Layer 2 steht
        #wiederholung bis zum letzten Layer, Werte seiner Neuronen werden zurückgegeben
        #Matrizen weights x werte werden multipliziert, vektor biases wird dazuaddiert
        #sigmoid reduziert Wertebereich auf Intervall 0-1
        a = [[z] for z in a]
        a = np.array(a)
        #print("Data shape:",a.shape)
        for i in range(len(self.weights)):
            a = sigmoid(np.dot(self.weights[i],a) + self.biases[i])
            #print("Data shape:",a.shape)
        return(a)


    def learn(self,input_data,output_data,mini_batch_size,epochs,learning_coefficient,input_test=None,output_test=None,test_stepsize=None):
        #input_data: array of lists of values for input layer neurons
        #output_data: array of desired output for each dataset
        #epochs: amount of times the learning process is repeated
        #mini_batch_size: amount of datasets that is used in every epoch
        #if test data is provided: testing among test data after test_stepsize epochs
        tStartTotal = time.time()
        for e in range(epochs):
            tStartEpoch = time.time()
            if epochs > 1:
                print("\nStarting epoch",e)
            #Shuffle Batch
            list_data = list(zip(input_data,output_data))
            random.shuffle(list_data)
            #subdivide into mini batches containing "mini_batch_size" datasets each
            mini_batches = [list_data[n*mini_batch_size:(n+1)*mini_batch_size] for n in range(len(list_data)//mini_batch_size)]
            pbs = []
            pws = []
            for mini_batch in mini_batches:
               pb,pw = self.gradient_descent(mini_batch,learning_coefficient)
               pbs.append(pb)
               pws.append(pw)            

            durEpoch = time.time()-tStartEpoch
            print("Epoch {} done after {} seconds.".format(e,durEpoch))          

            #evaluate, if test data is given
            if input_test:
                if e%test_stepsize == 0:
                    filep = open("data/graphs/deltaperc.txt","a")
                    new_percentage = self.evaluate(input_test,output_test)
                    line = "avg_pbs: {} avg_pws: {} delta_perc: {} to {}\n".format(np.array(pbs).mean(axis = 0),np.array(pws).mean(axis = 0),self.percentage,new_percentage)
                    self.percentage = new_percentage


            #print("Evaluation finished after {} seconds.".format(time.time() - tStartEpoch - durEpoch))
        if epochs > 1:
            print("Whole learning process finished after",round((time.time()-tStartTotal)/60*100)/100,"minutes.")


    def gradient_descent(self,mini_batch,learning_coefficient):
        #Tweaking weights/biases with minibatch by using backpropagation
        #learning coefficient: smaller coefficient results in more precise changes,
        #but therefore needs longer to adjust, default: 1
        delta_b = [np.zeros(bs.shape) for bs in self.biases]
        delta_w = [np.zeros(ws.shape) for ws in self.weights]

        #calculating gradient for every set of data using backpropagation
        for data_input,output_value in mini_batch:
            #adding gradient of one single set of data to total gradient
            delta_delta_b,delta_delta_w = self.backpropagation(data_input,output_value)
            delta_b = [db+ddb for db, ddb in zip(delta_b, delta_delta_b)]
            delta_w = [dw+ddw for dw, ddw in zip(delta_w, delta_delta_w)]
            #alternative solution that directly forms averages
            #delta_b = [db+ddb/len(mini_batch) for db, ddb in zip(delta_b, delta_delta_b)]
            #delta_w = [dw+ddw/len(mini_batch) for dw, ddw in zip(delta_w, delta_delta_w)]
            
        #calculating the average among sum of all gradients
        delta_b = [b/len(mini_batch) for b in delta_b]
        delta_w = [w/len(mini_batch) for w in delta_w]            

        #angeben, wie delta_b/delta_w als prozentualer anteil von b/w ist
        #pb = [np.average(abs(db))/np.average(abs(b)) for b,db in zip(self.biases,delta_b)]
        #pw = [np.average(abs(dw))/np.average(abs(w)) for w,dw in zip(self.weights,delta_w)]

        #reziproke (range: >1)
        pb = [np.average(abs(b))/np.average(abs(db)) for b,db in zip(self.biases,delta_b)]
        pw = [np.average(abs(w))/np.average(abs(dw)) for w,dw in zip(self.weights,delta_w)]

        #adding final negative gradient from networks weights/biases
        #print("weights: {}\ndelta_weights: {}\nbiases: {}\ndelta_biases: {}".format(self.weights,delta_w,self.biases,delta_b))
        
        self.weights = [w+dw*learning_coefficient for w, dw in zip(self.weights, delta_w)]
        self.biases = [b+db*learning_coefficient for b, db in zip(self.biases, delta_b)]

        return pb,pw

    def backpropagation(self,data_input,data_output):
        #calculating negative gradient for one set of data
        delta_b = [np.zeros(bs.shape) for bs in self.biases]
        delta_w = [np.zeros(ws.shape) for ws in self.weights]

        #alist: activations layer by layer speichern
        a = [[act] for act in data_input]
        alist = [a]
        zlist = []

        #feeden und activations/zs in listen speichern
        for i in range(len(self.weights)):
            z = np.dot(self.weights[i],a) + self.biases[i]
            zlist.append(z)
            a = sigmoid(z)
            alist.append(a)
        
        #erstes epsilon berechnen, damit letzte layers bei w/b errechnen
        epsilon = self.cost(alist[-1],data_output) * sigmoid_derivative(zlist[-1])
        delta_b[-1] = epsilon
        delta_w[-1] = np.dot(epsilon,alist[-2].transpose())

        for i in range(2,len(delta_b)+1):
            z = zlist[-i]
            #neues epsilon berechnen
            epsilon = np.dot(self.weights[-i+1].transpose(),epsilon) * sigmoid_derivative(z)
            delta_b[-i] = epsilon
            #print(alist[-(i+1)],"\n\n")
            delta_w[-i] = np.dot(epsilon,np.array(alist[-(i+1)]).transpose())
        return (delta_b,delta_w)

    def cost(self,output_neurons,output_value):
        theoretical_output = []
        for i in range(len(output_neurons)):
            if i == output_value:
                theoretical_output.append([1])
            else:
                theoretical_output.append([0])
        #print("theoretical output: {}, real output: {}".format(theoretical_output,output_neurons))
        diff = (np.array(theoretical_output) - output_neurons)
        #print("cost vector: {}, shape: {}".format(diff,diff.shape))
        return(diff)

    def evaluate(self,test_input,test_output,fname=None):
        correct = 0
        for i,o in zip(test_input,test_output):
            out = list(self.feed(i))
            n = out.index(max(out))
            #print("real value: {}, computed result: {}, value: {}".format(o,n,max(out)))
            if n == o:
                correct += 1
        print("{}/{} correct".format(correct,len(test_output)))

        #falls filename gegeben ist, in statsheet speichern
        if fname:
            i = fname.index("_")
            layers = fname[1:i]
            fname = fname[i+1:]
            i = fname.index("i")
            img = fname[0:i]
            fname = fname[i+4:]
            i = fname.index("b")
            mbs = fname[0:i]
            fname = fname[i+2:]
            i = fname.index("e")
            e = fname[:i]
            fname = fname[i+2:]
            i = fname.index("r")
            r = fname[:i]

            statsheet = open("data/statistics/stats.csv","a")
            statsheet.write("{};{};{};{};{};{};{}\n".format(layers,img,mbs,e,r,correct,len(test_output)))
            statsheet.close()
        return(correct/len(test_output))


    def saveNetwork(self,filename):
        #saved current layer sizes/weights/biases to a file
        print("saving to",filename)
        np.save("data/networks/files/"+filename,np.array([self.weights,self.biases],dtype = "object"))


    def loadNetwork(self,filename):
        data = np.load("data/networks/files/"+filename+".npy",allow_pickle=True)

        self.weights = data[0]
        self.biases = data[1]

        self.layer_sizes = [self.weights[0].shape[1]]
        for b in self.biases:
            self.layer_sizes.append(b.shape[0])


#andere wichtige Funktionen
def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_derivative(z):
    return (sigmoid(z)*(1-sigmoid(z)))