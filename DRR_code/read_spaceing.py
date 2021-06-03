# This program extract data from dicom file for patient 1


import os
from pydicom import*

# Print the current directory
cwd = os.getcwd()
print(cwd)

# Create a file to write data in
f = open("patient1_0004.txt", "w")

# To check single image
# dataset = dcmread('EXP0322')
# print(dataset)

# Print all images for patient 1
for k in range(1, 2247):
    if k <= 9:
        filename = 'EXP000%d' % k
        dataset = dcmread(filename)
        f.write("\n%s" % filename)
        print(filename)
        print(dataset)
        print("\n---------------------------------------------------------------------------------------------------\n")
    elif k <= 99:
        filename = 'EXP00%d' % k
        dataset = dcmread(filename)
        f.write("\n%s" % filename)
        print(filename)
        print(dataset)
        print("\n---------------------------------------------------------------------------------------------------\n")
    elif k <= 999:
        filename = 'EXP0%d' % k
        dataset = dcmread(filename)
        f.write("\n%s" % filename)
        print(filename)
        print(dataset)
        print("\n---------------------------------------------------------------------------------------------------\n")
    else:
        filename = 'EXP%d' % k
        dataset = dcmread(filename)
        f.write("\n%s" % filename)
        print(filename)
        print(dataset)
        print("\n---------------------------------------------------------------------------------------------------\n")

# Print the type of modality and write to file
    if 'US' in dataset.Modality:
        # f.write("%s" % filename, )
        f.write(",Ultrasound,")
    elif 'MR' in dataset.Modality:
        # f.write("\n%s" %filename,)
        f.write(",MRI,")
    else:
        # f.write("\n%s" %filename,)
        f.write(",Unknown,")

        # print("\n-------------------------------------------------------------------------------------------------\n")


# Print the image orientation
    # print("ImageOrientationPatient" in dataset)
    # if dataset.ImageOrientationPatient is not None:
    if "ImageOrientationPatient" in dataset:
        image_orientation = dataset.ImageOrientationPatient
        # print(dataset.ImageOrientationPatient)
        if image_orientation[0] > 0.8:
            """and image_orientation[0] < 1.1:"""
            round(image_orientation[0])
            # print(image_orientation[0])
            if image_orientation[4] > 0.8:
                """and image_orientation[4] < 1.1:"""
                round(image_orientation[4])
                # print(image_orientation[4])
                f.write("Axial")
            else:
                f.write("Coronal")
        else:
            f.write("Sagital")
    else:
        f.write("Not Provided")


# Print the slice thickness into the text file
    if "SliceThickness" in dataset:
        slice_thickness = dataset.SliceThickness
        f.write(",%d" % slice_thickness)
    else:
        f.write(",Not Provided")

# print("\n---------------------------------------------------------------------------------------------------------\n")


# Print the slice location into the text file
    if "SliceLocation" in dataset:
        slice_location = dataset.SliceLocation
        f.write(",%f" % slice_location)
    else:
        f.write(",Not Provided")

# print("\n---------------------------------------------------------------------------------------------------------\n")
    # Echo time
    if "EchoTime" in dataset:
        echo_time = dataset.EchoTime
        f.write(",%s" % echo_time)
    else:
        f.write(",Not Provided")

    # Repetition time
    if "RepetitionTime" in dataset:
        repetition_time = dataset.RepetitionTime
        f.write(",%s" % repetition_time)
    else:
        f.write(",Not Provided")

# Spacing Between Slices
    if "SpacingBetweenSlices" in dataset:
        spacing_between_slices = dataset.SpacingBetweenSlices
        f.write(",%d" % spacing_between_slices)
    else:
        f.write(",Not Provided")

f.close()
