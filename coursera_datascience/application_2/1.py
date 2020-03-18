import re
import numpy as np
from collections import Counter
from scipy.spatial.distance import cdist

file = open('sentences.txt')
text = list(file)
text = [i.lower() for i in text]
text = [re.split('[^a-z]', i) for i in text]

norm_text = []
words = []
for i in text:
    words += list(filter(None, i))
    norm_text.append(list(filter(None, i)))

dictionary = []
for i in norm_text:
    dictionary.append(Counter(i))

words = list(set(words))

matrix = np.array([[i[j] for j in words] for i in dictionary])

print (cdist(matrix[0:1], matrix[0:], metric='cosine'))
        



#print (words)


