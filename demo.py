
from src.data import load

arr = load._load_npz("mnist_mv")
print(arr.shape)