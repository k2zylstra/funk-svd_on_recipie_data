# # Funk SVD
# Kieran Zysltra

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

# SVD turns a matrix into three component matricies with the equation $A=UWV^T$ where $U$ is comprised of the eigenvectors of $AA^T$, $W$ is a diagonal matrix of the eigen values of $A^TA$ and $V^T$. with the funk svd method the rows and columns of $U$ and $V$ are not constrained, only the dot product of the two. Funks method changes this to $A=UV^T$ where $U$ is, in the case of netflix, a latent representation of movies, and $V^T$ is a latent representation of the users. Funk initializes this method with a random $U$ and a random $V$ and finds the gradient to train the model to closely aproximate the ranking a user gave to a specific movie.

# This project applys the Funk SVD method to a DNA VCF data. These files measure the allele frequency on mutations in the human genome as well as storing the rough location of each sample (Africa (AFR), East Asia (EAS), Europe (EUR), South Asia (SAS), and the Americas (AMR)). Each column in the file is a mutation gene and the value of the cell represents whether the gene is present or not. Since only about 1 mutation in 20 shows itself the data is very sparse. There are 4 different values that can be associated with the mutation. Recieved from the father, recieved from the mother, recieved from both, or recieved from neither. By applying the funk svd method on the DNA file we can try to predict whether a mutation will be present in the genenome of a person based off the general world region they are from. 

# +
U = tf.random.normal((3,3), dtype = 'float32')
V = tf.random.normal((3,3), dtype = 'float32')
 
#Our (3,3) "ranking" matrix.
M = U @ V
 
#we can visualize what our matrices look like



U_d = tf.Variable(tf.random.normal((3,3), dtype = 'float32'))
V_d = tf.Variable(tf.random.normal((3,3), dtype = 'float32'))
adam_opt = tf.keras.optimizers.Adam(learning_rate = 0.01)
 
epochs = 500
losses_adam = []
 
for ep in range(epochs):
 
    with tf.GradientTape() as tape:
        M_app = U_d @ V_d
        loss = tf.reduce_mean(tf.square(M - M_app))
 
    losses_adam.append(loss.numpy())
    grads = tape.gradient(loss, [U_d, V_d])
    adam_opt.apply_gradients(zip(grads, [U_d, V_d]))
 
plt.plot(losses_adam, label = 'Loss')
plt.legend()
plt.show()
# -

M_app = U_d @ V_d
plt.figure(figsize=(12,16))
plt.subplot(321)
plt.imshow(U)
plt.title('U original')
plt.subplot(322)
plt.imshow(U_d.numpy())
plt.title('U from tensorflow')
plt.subplot(323)
plt.imshow(V)
plt.title('V original')
plt.subplot(324)
plt.imshow(V_d.numpy())
plt.title('V from tensorflow')
plt.subplot(325)
plt.imshow(M)
plt.title('M original')
plt.subplot(326)
plt.imshow(M_app.numpy())
plt.title('M from tensorflow')
plt.show()



# +
U = tf.random.normal((500,500), dtype = 'float32')
V = tf.random.normal((500,500), dtype = 'float32')
M = U @ V
T = tf.zeros((500,500), dtype = 'float32')
for i in range(U.shape[0]):
    T = T + 1/(i + 1) * (U[:,i:i+1] @ V[i:i+1,:])
     
harm = np.array([1/(i + 1) for i in range(500) ])
plt.plot(harm)
plt.show()

# +

#adding a sparcity mask
sparcity_mat = np.ones((500,500))
 
#How much of the matrix to mask. This is not strictly kept as some random indices will coincide
sparcity_ratio = .7
i = np.random.randint(0, M.shape[0], int(sparcity_ratio * M.shape[0] * M.shape[1]))
j = np.random.randint(0, M.shape[1], int(sparcity_ratio * M.shape[0] * M.shape[1]))
sparcity_mat[i,j] = 0
 
sparcity_mat = tf.constant(sparcity_mat, dtype = 'float32')
print('nonzero entries ratio: ', (tf.reduce_sum(sparcity_mat)/(sparcity_mat.shape[0] * sparcity_mat.shape[1])).numpy())
 
#this matrix contains 1's where we have masked 
masked_entries = tf.cast(tf.not_equal(sparcity_mat, 1), dtype = 'float32')

# +
df = pd.read_csv("genome_data.csv")
sparcity_mat = df.to_numpy()
sparcity_mat = sparcity_mat[::,1:501]
sparcity_mat = sparcity_mat.astype(np.float32)

print(sparcity_mat.shape)
plt.figure(figsize=(10,10))
plt.imshow(sparcity_mat)
plt.show()


# -

def early_stopping(losses, patience = 5):
     
    if len(losses) <= patience + 1:
        return False
     
    avg_loss = np.mean(losses[-1 - patience:-1])
     
    if avg_loss - losses[-1] < 0.01*avg_loss:
        return True
     
    return False


# +
U_d = tf.Variable(tf.random.normal((500, 30)))
V_d = tf.Variable(tf.random.normal((30, 500)))
adam_opt = tf.keras.optimizers.Adam()

from datetime import datetime
ep = 0
start_time = datetime.now()
 
losses = []
val_losses = []
 
#normalization factors for training entries and validation entries.
train_norm = tf.reduce_sum(sparcity_mat)
val_norm = tf.reduce_sum(masked_entries)
 
while True:
     
    with tf.GradientTape() as tape:
        M_app = U_d @ V_d
         
        pred_errors_squared = tf.square(M - M_app)
        loss = tf.reduce_sum((sparcity_mat * pred_errors_squared)/train_norm)
         
    val_loss = tf.reduce_sum((masked_entries * pred_errors_squared)/val_norm)
 
    if ep%1000 == 0:
        print(datetime.now() - start_time, loss, val_loss, ep)
        losses.append(loss.numpy())
        val_losses.append(val_loss.numpy())
    if early_stopping(val_losses):
        break
     
    grads = tape.gradient(loss, [U_d, V_d])
    adam_opt.apply_gradients(zip(grads, [U_d, V_d]))
 
    ep += 1
 
print('total time: ', datetime.now() - start_time)
print('epochs: ', ep)

# +

T = tf.zeros((500,500), dtype = 'float32')
for i in range(U_d.shape[0]):
    T = T + 1/(i + 1) * (U_d[:,i:i+1] @ V_d[i:i+1,:])
     
harm = np.array([1/(i + 1) for i in range(500) ])
plt.plot(harm)
plt.show()
# -


