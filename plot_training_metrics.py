import matplotlib.pyplot as plt
import pickle

# Assuming history is already available after training, or load it if you saved it as a file
# For example, if you save `history.history` as a file, load it back like this:
# with open('history.pkl', 'rb') as file:
#     history = pickle.load(file)

# Plot accuracy
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.savefig('accuracy_plot.png')  # Save the accuracy plot

# Plot loss
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.savefig('loss_plot.png')  # Save the loss plot
