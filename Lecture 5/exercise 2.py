import torch
from math import log

sigmoid = torch.nn.Sigmoid()
relu = torch.nn.ReLU()
x1 = torch.tensor(1.0, requires_grad=True)
w1 = torch.tensor(3.0, requires_grad=True)

x2 = torch.tensor(2.0, requires_grad=True)
w2 = torch.tensor(4.0, requires_grad=True)

w_ho = torch.tensor(0.5, requires_grad=True)

y1 = x1 * w1
y1.register_hook(lambda grad: print("Grad y1 = {}".format(grad)))

y2 = x2 * w2
y2.register_hook(lambda grad: print("Grad y2 = {}".format(grad)))

y3 = relu(y1 + y2)
y3.register_hook(lambda grad: print("Grad y3 = {}".format(grad)))

y4 = y3 * w_ho
y4.register_hook(lambda grad: print("Grad y4 = {}".format(grad)))

y5 = sigmoid(y4)
y5.register_hook(lambda grad: print("Grad y5 = {}".format(grad)))

t = y4

e = -t * log(y5) - (1 - t) * log(1 - y5)

e.backward()

# print("Error e = {}".format(e))
print("Grad x1 = {}".format(x1.grad))
print("Grad x2 = {}".format(x2.grad))
print("Grad w1 = {}".format(w1.grad))
print("Grad w2 = {}".format(w2.grad))
print("Done")
