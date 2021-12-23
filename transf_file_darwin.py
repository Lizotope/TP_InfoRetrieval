import pickle5 as pickle
import sys
import re

# reading the data from the file
with open(sys.argv[1], 'rb') as handle:
    data = handle.read()
 
print("Data type before reconstruction : ", type(data))
  
# reconstructing the data as dictionary
d = pickle.loads(data)
  
print("Data type after reconstruction : ", type(d))
print(d['Charles Darwin'])


