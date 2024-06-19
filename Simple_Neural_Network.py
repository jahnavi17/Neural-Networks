import numpy as np

class SingleLayerNeuralNetwork:
    def __init__(self , input_size , output_size, learning_rate=0.01):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.weights = np.random.randn(input_size , output_size)
        self.bias = np.zeros((1,output_size))

    def sigmoid(self , z):
        return 1/(1 + np.exp(-z))
    
    def sig_derivative(self , z):
        return z * (1 - z)
    
    def forward(self , x) : 
        self.z = np.dot(x , self.weights) + self.bias
        self.a = self.sigmoid(self.z)

        return self.a 
    
    def backward(self , x , y , output):
        m = x.shape[0]
        error = output - y

        weights_update = np.dot(x.T , error * self.sig_derivative(output)) / m
        bias_update = np.sum(error * self.sig_derivative(output)) / m

        self.weights = self.learning_rate * weights_update
        self.bias = self.learning_rate * bias_update

    def train(self, x , y , epochs):
         for epoch in range(epochs) : 
             output = self.forward(x)
             self.backward(x , y , output)

             if (epoch % 100)==0 : 
                 loss = np.mean((y-output) ** 2)
                 print(f"Epoch : {epoch} , Loss : {loss}")

    def predict(self , x):
        output = self.forward(x)
        return np.round(output)

if __name__ == "__main__" :

    x = np.random.randn(100,2)
    y = np.round(np.random.randn(100,1))

    sample_nn = SingleLayerNeuralNetwork(input_size=2 , output_size=1, learning_rate=0.01)
    sample_nn.train(x , y , epochs=1000)
    
    test_input = np.array([[0.1, 0.2]])
    print(f"Test Input: {test_input}, Predicted Output: {sample_nn.predict(test_input)}")


