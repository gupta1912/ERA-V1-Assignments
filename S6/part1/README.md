# Backpropagation explanation

## Backpropagation: Step wise calculations
 It's a two layer neural network. Here we use sigmoid function as a activation fucntion(__σ__). It includes two input neurons (i1,i2), two hidden layer neurons (h1,h2), two output layer neurons and total 
 eight weights(w1,w2,w3,w4......,w8) is used between input to output layer. Learning rate __η__ is used to calculate the weight updates.
 
 **[Backpropagation sheet](https://docs.google.com/spreadsheets/d/1vWviA3FWfI7j4NWyteXR-7jjkCc0yH7g/edit?usp=sharing&ouid=102840136744817488430&rtpof=true&sd=true)**



  <img width="1007" alt="Screenshot 2023-06-07 at 8 25 52 AM" src="https://github.com/Tulsi97/ERAV1_dev/assets/35035797/0c258b3a-eb1b-4af0-81bf-1dbe870147ce">
  
  
### Block - 1 : 
- Calculation provided below performs the forward pass of the neural network, where input values are propagated through the hidden layer to the output layer, and the final output values are computed. The activation function introduces non-linearity, and the error is calculated to assess the performance of the network.
     
      1 - h1 = i1*w1 + i2*w2  =>  calculates the weighted sum of the input values (i1 and i2) with their corresponding weights (w1 and w2) for 1st neuron of hidden layer.
      2 - h2 = i1*w3 + i2*w4  =>  calculates the weighted sum of the input values (i1 and i2) with their corresponding weights (w3 and w4) for 2nd neuron of hidden layer.
      3 - a_h1 = σ(h1)        =>  applied the activation function(σ)to the activations of the hidden layer neurons, producing the output values a_h1 and a_h2.
      4 - a_h2 = σ(h2)
      5 - o1 = a_h1*w5 + a_h2*w6  =>  calculate the weighted sum of the hidden layer outputs (a_h1 and a_h2) with their corresponding weights (w5, w6, w7, and w8) to compute the activations of the output layer neurons (o1 and o2) 
      6 - o2 = a_h1*w7 + a_h2*w8
      7 - a_o1 = σ(o1)        =>  applied the activation function(σ)to the activations of the output layer neurons, producing the output values a_o1 and a_o2
      8 - a_o2 = σ(o2)
      9 - E1 = 1/2(t1 - a_o1)^2    =>  calculate the individual errors (squared differences) between the target values (t1 and t2) and the corresponding output values (a_o1 and a_o2)
      10 - E2 = 1/2(t2 - a_o2)^2
      11 - E_total = E1+E2      =>   This line computes the total error by summing up the individual errors E1 and E2


### Block - 2 : 
- This block performs the partial derivative of the total error with respect to w5, taking into account the individual error E1, the output activation a_o1, the sigmoid derivative ∂a_o1/∂o1, and the hidden layer activation a_h1. This derivative is crucial for updating the weights during the backpropagation phase of the neural network training.

       1 - ∂E_total/∂w5 = ∂(E1+E2)/∂w5          => Line(1,2) states that the derivative of the total error with respect to weight w5 is equal to the derivative of the sum of individual errors (E1 and E2) with respect to w5 but E2 is independent from w5, so the total error is equal to the partial derivative of E1 wrt w5.
       2 - ∂E_total/∂w5 = ∂E1/∂w5
       3 - ∂E_total/∂w5 = ∂E1/∂a_o1 * ∂a_o1/∂o1 * ∂o1/∂w5      => This line applies the chain rule of derivatives to further break down ∂E1/∂w5. It expresses the derivative of E1 with respect to w5 as the product of three partial derivatives: ∂E1/∂a_o1, ∂a_o1/∂o1, and ∂o1/∂w5.
       4 - ∂E1/∂a_o1 = ∂(1/2(t1 - a_o1)^2)/∂a_o1 = a_o1 - t1     =>  This line calculates the derivative of E1 with respect to a_o1, which is the difference between the output value a_o1 and the target value t1.
       5 - ∂a_o1/∂o1 =  ∂(σ(o1))/∂o1 = a_o1 * (1 - a_o1)  => This line computes the derivative of the activation function σ applied to o1 with respect to o1. It is equivalent to the output value a_o1 multiplied by (1 - a_o1). And the derivative for the sigmoid activation function is σ(x)*(1-σ(x)).
       6 - ∂o1/∂w5 = a_h1    => We can clearly see from block 1 derivative of o1 which is eual to (a_h1*w5 + a_h2*w6) wrt w5, is a_h1.
       
       
       
 ### Block - 3 : 
- These equations are derived using the chain rule of derivatives and the backpropagation algorithm. They are used to compute the gradients of the weights w5, w6, w7, and w8 during the training process. By updating the weights in the opposite direction of their gradients, the network learns to minimize the error and improve its performance.

       1 - ∂E/∂w5 = (a_o1-t1) * a_o1 * (1-a_o1) * a_h1   =>  This line calculates the partial derivative of the error E with respect to weight w5. It represents the contribution of weight w5 to the overall error and is used to update the weight during the backpropagation process. The expression on the right-hand side consists of terms involving the output activation a_o1, the target value t1, the sigmoid derivative (1-a_o1), and the hidden layer activation a_h1. Same calculations is performs for rest of the weights w6,w7,w8.
       2 - ∂E/∂w6 = (a_o1-t1) * a_o1 * (1-a_o1) * a_h2
       3 - ∂E/∂w7 = (a_o2-t2) * a_o2 * (1-a_o2) * a_h1
       4 - ∂E/∂w8 = (a_o2-t2) * a_o2 * (1-a_o2) * a_h2
       
       
### Block - 4 :
- The given equations represent the partial derivatives of the error function with respect to the activations of the hidden layer in a two-layer neural network. Specifically, ∂E1/∂a_h1 and ∂E2/∂a_h1 denote the contributions of a_h1 to the errors E1 and E2, respectively, while ∂E1/∂a_h2 and ∂E2/∂a_h2 represent the contributions of a_h2 to the same errors. These contributions are calculated by multiplying the terms (a_o1 - t1) and (a_o2 - t2), which measure the discrepancies between the output activations and their corresponding target values, with the terms a_o1, a_o2, (1 - a_o1), and (1 - a_o2), which capture the behavior of the sigmoid activation function. Additionally, the weights w5, w6, w7, and w8 are involved in the calculations to reflect the impact of the connections between the hidden and output layers. Finally, the expressions for ∂E_total/∂a_h1 and ∂E_total/∂a_h2 represent the overall contributions of a_h1 and a_h2, respectively, to the total error E_total, taking into account the contributions from both E1 and E2.

      1 - ∂E1/∂a_h1 = ∂E/∂a_o1 * ∂a_o1/∂o1 *∂o1/a_h1
      2 - ∂E1/∂a_h1 = (a_o1 - t1) * a_o1 * (1 - a_o1) * w5
      3 - ∂E2/∂a_h1 = (a_o2 - t2) * a_o2 * (1 - a_o2) * w7
      4 - ∂E1/∂a_h2 = (a_o1 - t1) * a_o1 * (1 - a_o1) * w6
      5 - ∂E2/∂a_h2 = (a_o2 - t2) * a_o2 * (1 - a_o2) * w8
      6 - ∂E_total/∂a_h1 = (a_o1 - t1) * a_o1 * (1 - a_o1) * w5 +  (a_o2 - t2) * a_o2 * (1 - a_o2) * w7
      7 - ∂E_total/∂a_h2 = (a_o1 - t1) * a_o1 * (1 - a_o1) * w6 +  (a_o2 - t2) * a_o2 * (1 - a_o2) * w8
      

### Block - 5 :
- The given equations represent the partial derivatives of the total error with respect to the weights in a two-layer neural network. ∂E_total/∂w1 and ∂E_total/∂w2 denote the contributions of w1 and w2, respectively, to the total error, while ∂E_total/∂w3 and ∂E_total/∂w4 represent the contributions of w3 and w4, respectively. These contributions are calculated by multiplying the terms ∂E_total/∂a_h1 and ∂E_total/∂a_h2, which measure the contributions of the hidden layer activations to the total error, with the terms ∂a_h1/∂h1 and ∂a_h2/∂h2, which capture the derivatives of the activation functions with respect to the weighted sums at the hidden layer. Finally, the expressions for ∂h1/∂w1, ∂h1/∂w2, ∂h2/∂w3, and ∂h2/∂w4 represent the derivatives of the weighted sums at the hidden layer with respect to the corresponding weights. By calculating these partial derivatives, the overall contributions of the weights to the total error can be determined and used for weight updates during the training process.

      1 - ∂E_total/∂w1 = ∂E_total/∂a_h1 * ∂a_h1/∂h1 * ∂h1/∂w1
      2 - ∂E_total/∂w2 = ∂E_total/∂a_h1 * ∂a_h1/∂h1 * ∂h1/∂w2
      3 - ∂E_total/∂w3 = ∂E_total/∂a_h2 * ∂a_h2/∂h2 * ∂h2/∂w3
      4 - ∂E_total/∂w4 = ∂E_total/∂a_h2 * ∂a_h2/∂h2 * ∂h2/∂w4


### Block - 6 :
- These equations provided represent the partial derivatives of the total error with respect to the weights in a two-layer neural network. ∂E_total/∂w1, ∂E_total/∂w2, ∂E_total/∂w3, and ∂E_total/∂w4 denote the contributions of the corresponding weights to the total error. These contributions are calculated by multiplying the terms ((a_o1 - t1) * a_o1 * (1 - a_o1) * w5 + (a_o2 - t2) * a_o2 * (1 - a_o2) * w7) and ((a_o1 - t1) * a_o1 * (1 - a_o1) * w6 + (a_o2 - t2) * a_o2 * (1 - a_o2) * w8) with the terms a_h1 * (1 - a_h1) and a_h2 * (1 - a_h2), respectively, which represent the derivatives of the hidden layer activations with respect to the weighted sums at the hidden layer. Finally, these products are multiplied by i1 and i2, which represent the input values corresponding to the respective weights. By evaluating these partial derivatives, the contributions of the weights to the total error can be determined, allowing for weight updates during the training process.

      1 - ∂E_total/∂w1 = ((a_o1 - t1) * a_o1 * (1 - a_o1) * w5 +  (a_o2 - t2) * a_o2 * (1 - a_o2) * w7) * a_h1 * (1 - a_h1) * i1
      2 - ∂E_total/∂w2 = ((a_o1 - t1) * a_o1 * (1 - a_o1) * w5 +  (a_o2 - t2) * a_o2 * (1 - a_o2) * w7) * a_h1 * (1 - a_h1) * i2
      3 - ∂E_total/∂w3 = ((a_o1 - t1) * a_o1 * (1 - a_o1) * w6 + (a_o2 - t2) * a_o2 * (1 - a_o2) * w8) * a_h2 * (1 - a_h2) * i1
      4 - ∂E_total/∂w4 = ((a_o1 - t1) * a_o1 * (1 - a_o1) * w6 +  (a_o2 - t2) * a_o2 * (1 - a_o2) * w8) * a_h2 * (1 - a_h2) * i2


## The error graphs at different learning rate : [0.1, 0.2, 0.5, 0.8, 1.0, 2.0] 
   - learning rate = 0.1

     ![image](https://github.com/Tulsi97/ERAV1_dev/assets/35035797/ef68bc23-996d-4217-b8ca-46caa52a549b)
     
 
   - learning rate = 0.2

     ![image](https://github.com/Tulsi97/ERAV1_dev/assets/35035797/bf96dbac-d726-4efb-b22b-daf2afda0fb1)
     
   - learning rate = 0.5
   
     ![image](https://github.com/Tulsi97/ERAV1_dev/assets/35035797/95848f80-81aa-4c1f-ba0f-e84a297a8920)


   - learning rate = 0.8
   
     ![image](https://github.com/Tulsi97/ERAV1_dev/assets/35035797/cb538ee2-5f62-41ec-b22f-d18ca0b400bd)


   - learning rate = 1
  
     ![image](https://github.com/Tulsi97/ERAV1_dev/assets/35035797/058cff32-fddc-4497-bb6a-f0753a62645b)


   - learning rate = 2
 
     ![image](https://github.com/Tulsi97/ERAV1_dev/assets/35035797/331cdfcf-4158-4463-98ab-33e8043902a6)










