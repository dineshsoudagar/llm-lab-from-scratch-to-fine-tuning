import math

import torch


class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.A = torch.nn.Parameter(torch.empty(in_dim, rank))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))  # similar to standard weight initialization
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x


"""
In the code above, rank is a hyperparameter that controls the inner dimension of the matrices 
 and 
In other words, this parameter controls the number of additional parameters introduced by LoRA and is a key factor in 
determining the balance between model adaptability and parameter efficiency
The second hyperparameter, alpha, is a scaling hyperparameter applied to the output of the low-rank adaptation
It essentially controls the extent to which the adapted layer's output is allowed to influence the original output of 
the layer being adapted
This can be seen as a way to regulate the impact of the low-rank adaptation on the layer's output
So far, the LoRALayer class we implemented above allows us to transform the layer inputs 
However, in LoRA, we are usually interested in replacing existing Linear layers so that the weight update is applied to 
the existing pretrained weights.
To incorporate the original Linear layer weights as shown in the figure above, we implement a LinearWithLoRA layer below
that uses the previously implemented LoRALayer and can be used to replace existing Linear layers in a neural network, 
for example, the self-attention module or feed forward modules in an LLM
"""


class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear  # Original model weights that will be frozen
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha  # new layer that are updated during training
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)  # this self.lora is the weights that are trained to modify
        # the model behavior on new input data upon fine-tuning.

"""
Note that since we initialize the weight matrix 
 (self.B in LoRALayer) with zero values in the LoRA layer, the matrix multiplication between 
 and 
 results in a matrix consisting of 0's and doesn't affect the original weights (since adding 0 to the original weights does not modify them)
"""
