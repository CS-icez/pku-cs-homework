import numpy as np
import cv2
import os


def read_img(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


if __name__ == "__main__":

    #input
    input_vector = np.zeros((10,784)) 
    for i in range(10):
        input_vector[i,:] = read_img("mnist_subset/"+str(i)+".png").reshape(-1)/255.
    gt_y = np.zeros((10,1)) 
    gt_y[0] =1  

    np.random.seed(14)

    #Intialization MLP  (784 -> 16 -> 1)
    MLP_layer_1 = np.random.randn(784,16)
    MLP_layer_2 = np.random.randn(16,1)
    lr=1e-1
    loss_list=[]

    for i in range(50):
        #Forward
        output_layer_1 = input_vector.dot(MLP_layer_1)
        output_layer_1_act = 1 / (1 + np.exp(-output_layer_1))  #sigmoid activation function
        output_layer_2 = output_layer_1_act.dot(MLP_layer_2)
        pred_y = 1 / (1 + np.exp(-output_layer_2))  #sigmoid activation function
        loss = -(gt_y * np.log(pred_y) + (1 - gt_y) * np.log(1 - pred_y)).sum() #cross-entroy loss
        print("iteration: %d, loss: %f" % (i + 1 ,loss))
        loss_list.append(loss)


        # Backward : compute the gradient of paratmerters of layer1 (grad_layer_1) and layer2 (grad_layer_2)

        grad_output_layer_2 = pred_y - gt_y
        grad_layer_2 = output_layer_1_act.T.dot(grad_output_layer_2)
        grad_output_layer_1_act = grad_output_layer_2.dot(MLP_layer_2.T)
        grad_output_layer_1 = grad_output_layer_1_act * (output_layer_1_act * (1 - output_layer_1_act))
        grad_layer_1 = input_vector.T.dot(grad_output_layer_1)

        MLP_layer_1 -= lr * grad_layer_1
        MLP_layer_2 -= lr * grad_layer_2

    os.makedirs("results", exist_ok=True)
    np.savetxt("results/BP.txt", loss_list)