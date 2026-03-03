from MyTorch.scalar import Scalar
from MyTorch.datasets import simple
from MyTorch.module import Module
import random


class SimpleModel():
    """A simple linear classifier: y= sigmoid(w*x+b)"""
    def __init__(self):
        # Initialize parameters
        self.w:"Scalar"=Scalar(random.uniform(-1,1))
        self.w.requires_grad_(True)
        self.b:"Scalar"=Scalar(random.uniform(-1,1))
        self.b.requires_grad_(True)
    
    def forward(self,x:float)->Scalar:
        """Predict probability for input x."""
        return (self.w*x+self.b).sigmoid()

    def parameters(self):
        """Return all trainable parameters."""
        return [self.w,self.b]
    
    def zero_grad(self):
        """Reset all gradients to None."""
        for p in self.parameters():
            p.zero_grad_()


def binary_cross_entropy(pred:Scalar,target:float)->Scalar:
    """
    Compute binary cross-entropy loss.
    loss=-[y*log(p)+(1-y)*log(1-p)]
    """

    eps=1e-7 #Numerical stability
    pred_clamped=Scalar(max(eps,min(1-eps,pred.data)))
    pred_clamped.requires_grad_(True)
    pred.requires_grad_(True)

    if target==1:
        return -pred.log()
    else:
        return -(Scalar(1.0)-pred).log()

def train_sample():
    """Train on the simple dataset"""

    # Generate data.
    X,y=simple(100)

    # Create model
    model=SimpleModel()
    learning_rate=0.5

    for epoch in range(65):
        total_loss=0.0
        correct=0
        for (x_coord,_),label in zip(X,y):
            # Zero gradients.
            model.zero_grad()

            # Forward pass
            pred=model.forward(x_coord)

            # Compute loss
            loss=binary_cross_entropy(pred,label)
            total_loss+=loss.data

            loss.backward()


            ### Update parameters (SGD)
            for param in model.parameters():
                if param.derivative is not None:
                    param.data=param.data - learning_rate*param.derivative
            model.zero_grad()


            # Tracking accuracy
            predicted_label=1 if pred.data>0.5 else 0
            if predicted_label==label:
                correct+=1
            
        # if epoch%10==0:
        accuracy=correct/len(X)
        print(f"Epoch {epoch}: loss={total_loss/len(X):.4f}, accuracy={accuracy:.2f}%")

        
    print(f"\nFinal Parameters: w={model.w.data:.4f}, b={model.b.data:.4f}")
    print(f"Expected decision boundary: x = 0.5")
    print(f"Learned boundary: x = {-model.b.data/model.w.data:.4f}")

if __name__=="__main__":
    train_sample()