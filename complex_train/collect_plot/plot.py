import matplotlib.pyplot as plt
import numpy as np

# Load data from text file using numpy
data = np.loadtxt('output_result_x.txt')
#epoch:  rmse:   mse:    pearson spearman        ci

import numpy as np

aspect_ratio = 2.5
width = 8
height = width / aspect_ratio
fig = plt.figure(figsize=(width, height))


# Panel 1
plt.subplot(2, 3, 1)
plt.plot(data[:, 0], data[:, 1])
plt.xlabel('Epoch')
plt.ylabel('AUC')

# Panel 2
plt.subplot(2, 3, 2)
plt.plot(data[:, 0], data[:, 2])
plt.xlabel('Epoch')
plt.ylabel('TPR')
# Panel 3
plt.subplot(2, 3, 3)
plt.plot(data[:, 0], data[:, 3])
plt.xlabel('Epoch')
plt.ylabel('Precision')
# Panel 4
plt.subplot(2, 3, 4)
plt.plot(data[:, 0], data[:, 4])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

# Panel 5
plt.subplot(2, 3, 5)
plt.plot(data[:, 0], data[:, 5])
plt.xlabel('Epoch')
plt.ylabel('MCC')
# Adjust spacing between subplots



plt.subplots_adjust(wspace=0.5, hspace=1.2,bottom=0.25)

# Display the plot
#plt.show()

plt.savefig('output.png', dpi=300)

