

import os
arr = os.listdir('D:/research/RESULT_CHEST_CT_gamma_correction/RESULT_x/Patient1/')
print(arr)
f = open("dataset.txt", "a")
for filname in arr:
    m='D:/research/RESULT_CHEST_CT_gamma_correction/RESULT_x/Patient1/'+filname
    m=m.replace("/", "\\")


    f.write(m)
    f.write('\n')
f.close()