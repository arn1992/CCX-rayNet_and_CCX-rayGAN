import os

#p= os.listdir(os.getcwd())
#print(p)


# Function to rename multiple files

def main():
    i = 1
    j=80


    d="D:/polynomial/covid19/CT/Others/Patient ({})/".format(j)

    for filename in os.listdir(d):
        dst = "Others" + str(i)+'.png'
        src = d + filename
        dst = d + dst

        # rename() function will
        # rename all the files
        os.rename(src, dst)
        i += 1


# Driver Code
if __name__ == '__main__':
    # Calling main() function
    main()
