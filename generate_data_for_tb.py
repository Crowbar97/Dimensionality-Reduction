import os

import numpy as np

import pandas as pd

import tensorflow as tf
from tensorboard.plugins import projector


# reading from csv file
images = pd.read_csv('mnist_test.csv', names=['label'] + list(range(784)))


# converting to numpy arrays
X = images.iloc[:, 1:].to_numpy()
X = X / 255.0
y = images.iloc[:, 0].to_numpy()


# Set up a logs directory, so Tensorboard knows where to look for files
PATH = os.getcwd()
log_dir=os.path.join(PATH, 'tensorboard_logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Save Labels separately on a line-by-line manner.
with open(os.path.join(log_dir, 'metadata.tsv'), "w") as f:
    for label in y:
        f.write("{}\n".format(label))

# Save the weights we want to analyse as a variable. Note that the first
# value represents any unknown word, which is not in the metadata, so
# we will remove that value.
weights = tf.Variable(pd.DataFrame(X).values)

# Create a checkpoint from embedding, the filename and key are
# name of the tensor.
checkpoint = tf.train.Checkpoint(embedding=weights)
checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))

# Set up config
config = projector.ProjectorConfig()
embedding = config.embeddings.add()

# The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`
embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
embedding.metadata_path = 'metadata.tsv'
projector.visualize_embeddings(log_dir, config)
