import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

c = 0
row = [] 
m = []
with open("genome_data.vcf") as fpgenome:
    for l in fpgenome:
        words = l.split("\t")
        c+=1
        for w in words:
            if w == "0|0":
                row.append(0)
            elif w == "1|0":
                row.append(1)
            elif w == "0|1":
                row.append(1)
            elif w == "1|1":
                row.append(1)
        m.append(row)
        row = []
        if c == 500:
            break

df = pd.DataFrame(m, index=None, columns=None)
df.to_csv("genome_data.csv")
print(m)