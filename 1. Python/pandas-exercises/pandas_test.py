import pandas as pd
import numpy as np
df2 = pd.DataFrame(np.random.random(size=(5, 10)), columns=list('abcdefghij'))
cols_sum = df2.sum(axis=0)
print(cols_sum)
lst_of_key = list(dict(cols_sum).keys())
lst_of_val = list(dict(cols_sum).values())
index_of_min_value = lst_of_val.index(min(cols_sum))
print(lst_of_key[index_of_min_value])