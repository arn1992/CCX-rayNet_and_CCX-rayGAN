import numpy as np
import matplotlib.pyplot as plt
import os
import pydicom
import scipy.ndimage
from plotly.tools import FigureFactory as FF
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
def load_scan(path):
    slices = [pydicom.read_file(os.path.join(path, s)) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness


    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)
    return slices, image

def get_pixels_hu(image, slice_list):
    image[image == -2000] = 0


    for slice_number in range(len(slice_list)):

        intercept = slice_list[slice_number].RescaleIntercept
        slope = slice_list[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)

INPUT_FOLDER = 'D:/research/done/'
patients = os.listdir(INPUT_FOLDER)
patients.sort()
pat=[]
for patient in patients[0:3]:
    print(patient)
    slice_list, image = load_scan(os.path.join(INPUT_FOLDER, patient))
    print(type(slice_list))
    pat=slice_list
    print("slice thickness: ",slice_list[0].SliceThickness)
    print("Pixel Spacing (row, col): (%f, %f) " % (slice_list[0].PixelSpacing[0], slice_list[0].PixelSpacing[1]))
    image = get_pixels_hu(image, slice_list)
    #print(image)
    #np.save(os.path.join('3D_ARRAY_KUN', 'CT_3D_{}.npy'.format(patient)), image)


path1='D:/polynomial/resize_image/3D_ARRAY_KUN/CT_3D_Patient14.npy'
path2='D:/research/3D_ARRAY/CT_3D_(14).npy'
imgs_to_process1=np.load(path1)
imgs_to_process2=np.load(path2)
def sample_stack(stack, rows=6, cols=6, start_with=10, show_every=2):
    fig,ax = plt.subplots(rows,cols,figsize=[12,12])
    for i in range(rows*cols):
        ind = start_with + i*show_every
        ax[int(i/rows),int(i % rows)].set_title('slice %d' % ind)
        ax[int(i/rows),int(i % rows)].imshow(stack[ind],cmap='gray')
        ax[int(i/rows),int(i % rows)].axis('off')
    plt.show()

sample_stack(imgs_to_process1)
sample_stack(imgs_to_process2)




def resample(image, scan, new_spacing=[1, 1, 1]):
    # Determine current pixel spacing
    spacing = map(float, ([scan[0].SliceThickness] + list(scan[0].PixelSpacing)))
    spacing = np.array(list(spacing))

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)

    return image, new_spacing


print("Shape before resampling", imgs_to_process1.shape)
imgs_after_resamp, spacing = resample(imgs_to_process1, pat, [1,1,1])
print ("Shape after resampling", imgs_after_resamp.shape)




def make_mesh(image, threshold=-300, step_size=1):
    print("Transposing surface")
    p = image.transpose(2, 1, 0)

    print("Calculating surface")
    verts, faces, norm, val = measure.marching_cubes_lewiner(p, threshold, step_size=step_size, allow_degenerate=True)
    return verts, faces


def plotly_3d(verts, faces):
    x, y, z = zip(*verts)

    print ("Drawing")

    # Make the colormap single color since the axes are positional not intensity.
    #    colormap=['rgb(255,105,180)','rgb(255,255,51)','rgb(0,191,255)']
    colormap = ['rgb(236, 236, 212)', 'rgb(236, 236, 212)']

    fig = FF.create_trisurf(x=x,
                            y=y,
                            z=z,
                            plot_edges=False,
                            colormap=colormap,
                            simplices=faces,
                            backgroundcolor='rgb(64, 64, 64)',
                            title="Interactive Visualization")
    iplot(fig)


def plt_3d(verts, faces):
    print("Drawing")
    x, y, z = zip(*verts)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], linewidths=0.05, alpha=1)
    face_color = [1, 1, 0.9]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, max(x))
    ax.set_ylim(0, max(y))
    ax.set_zlim(0, max(z))
    #ax.set_axis_bgcolor((0.7, 0.7, 0.7))
    ax.set_facecolor("black")
    plt.savefig("3d1.png")
    plt.show()

v, f = make_mesh(imgs_after_resamp, 350)
plt_3d(v, f)

#v, f = make_mesh(imgs_after_resamp, 350, 2)
#plotly_3d(v, f)