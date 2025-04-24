import matplotlib.pyplot as plt

# Simulated accuracy data across epochs for different learning rates
epochs = list(range(1, 6))
lr_1e5 = [0.6221, 0.6453, 0.5988, 0.6512, 0.6512] 
lr_3e5 = [0.5816, 0.5673, 0.5699, 0.59, 0.5677]  
lr_5e6 = [0.5854, 0.6052, 0.6147, 0.6124, 0.6188]  

plt.plot(epochs, lr_3e5, label='lr=3e-5', marker='o')
plt.plot(epochs, lr_1e5, label='lr=1e-5', marker='o')
plt.plot(epochs, lr_5e6, label='lr=5e-6', marker='o')

plt.title("Validation Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim(0.50, 0.70)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
