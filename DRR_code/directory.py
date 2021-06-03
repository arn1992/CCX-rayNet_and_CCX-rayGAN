import os
import cv2
x='D:/research/Chest_CT_Dataset_JPG/Patient-/Study__CT/'
output = [dI for dI in os.listdir(x) if os.path.isdir(os.path.join(x,dI))]
print(output[0])

num=1
for i in range(len(output)):

    datapath='D:/research/Chest_CT_Dataset_JPG/Patient-/Study__CT/{}/'.format(output[i])
    print(datapath)

    files_list = os.listdir(datapath)
    img_sum = 0
    files_list = files_list[0:128]

    for file in files_list:
        img = cv2.imread(datapath + file).astype('float64')
        img_sum += img

    img_sum = img_sum / len(files_list)

    cv2.imwrite('D:/polynomial/CT/CHEST_CT_XRAY/Xray_(%d).jpg'%num, img_sum)
    num=num+1



#plot.savefig('hanning(%d).pdf' % num)
