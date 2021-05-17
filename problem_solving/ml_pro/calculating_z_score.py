
#!/usr/bin/python3

import numpy as np



# Please do not change the below function name and parameters
def normalizer(list_of_num):
    l = np.asarray(list_of_num)
    
    mean = np.mean(l)

    sd = np.sqrt((np.power((l - mean), 2) / (len(l) -1 ) ))
    
    print((l[-1] - mean)/sd)



