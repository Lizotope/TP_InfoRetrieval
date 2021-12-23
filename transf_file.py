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
print(d)


#search the max value
all_values = d.values()
max_value = max(all_values)
print("la val max est : ", max_value)
# search the key of max value
max_key = max(d, key=d.get)
print("la key de la val max est : ", max_key)

