import torch 

# my_tensor = torch.tensor([[1,2,3],[4,5,6]])

# we can mention the data types and set cuda too!

device = "cuda" if torch.cuda.is_available() else "cpu"

# my_tensor2 = torch.tensor([[1,2,3],[4,5,6]],dtype=torch.float32,device="cuda",requires_grad=True)


# print(my_tensor)
# print(my_tensor2)
# print(my_tensor2.dtype)
# print(my_tensor2.device)
# print(my_tensor2.shape)
# print(my_tensor2.requires_grad)


# # Other common initialisation methods

# x = torch.empty(size=(3,3))
# print(x)
# y = torch.zeros((3,3))
# print(y)

# z = torch.rand((3,3))

# print(z)

# c = torch.eye(4,4) # identity matrix or tensor
# print(c)

# d = torch.arange(start=0,end=5,step=1)
# print(d)

# e = torch.linspace(start=0.1,end=1,steps=10)
# print(e)

# f = torch.empty(size=(1,5)).normal_(mean=0,std=1)
# print(f)

# g = torch.empty(size=(1,5)).uniform_(0,1)
# print(g)

# h = torch.diag(torch.ones(3))
# print(h)

# # How to initialise and convert tensors to other types ( int , float, double)

tensor = torch.arange(4) #thats the end of the range this prints for 0 to 3

print(tensor.bool()) #prints the boolean values of the tensor 
print(tensor.short())
print(tensor.long()) #int64
print(tensor.half()) #float16
print(tensor.float()) #float32

