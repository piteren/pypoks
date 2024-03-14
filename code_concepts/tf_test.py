"""

 2020 (c) piteren

"""
import tensorflow as tf

for _ in range(10):
    print(tf.Graph())

print()

for _ in range(10):
    g = tf.Graph()
    print(g)

print()

graphs = [tf.Graph() for _ in range(10)]
for g in graphs:
    print(g)