import os
import sys
import getopt
import pickle
import nibabel as nib
import numpy as np
import cv2


# set working directory
PATH = 'D:/machine_learning/gm_segmentation_dti'
PATH_test = 'D:/machine_learning/gm_segmentation_dti/test'
os.chdir(PATH)

# get the arguments from the command-line except the filename
argv = sys.argv[1:]
print('ARGV      :', sys.argv[1:])
try:
    # Define the getopt parameters
    options, remainder = getopt.getopt(argv, '', ['fa=','md=','ad=','rd=','mask_sc='])
    print('OPTIONS   :', options)
    # Check if the options' length is 2 (can be enhanced)
    if len(options) == 0 and len(options) > 5:
      print ('usage: segment_dti.py --fa <fa map> --md <md map> --ad <ad map> --rd <rd map> --mask <mask>')
    else:
      # Iterate the options and get the corresponding values
      for opt, arg in options:
          if opt == "--fa":
              fa_fname = arg
          elif opt == "--md":
              md_fname = arg
          elif opt == "--ad":
              ad_fname = arg
          elif opt == "--rd":
              rd_fname = arg
          elif opt == "--mask_sc":
              mask_sc_fname = arg

except getopt.GetoptError:
    # Print something useful
    print('usage: segment_dti.py --fa <fa map> --md <md map> --ad <ad map> --rd <rd map> --mask_sc <sc mask>')
    sys.exit(2)

#fa_fname = os.path.join(PATH_test,'fa.nii')
#md_fname = os.path.join(PATH_test,'md.nii')
#ad_fname = os.path.join(PATH_test,'ad.nii')
#rd_fname = os.path.join(PATH_test,'rd.nii')
#mask_fname = os.path.join(PATH_test,'mask_sc.nii')

# load in the four dti maps (FA, MD, AD, RD) to segment
fa_obj = nib.load(fa_fname)
md_obj = nib.load(md_fname)
ad_obj = nib.load(ad_fname)
rd_obj = nib.load(rd_fname)

fa = fa_obj.get_fdata()
md = md_obj.get_fdata()
ad = ad_obj.get_fdata()
rd = rd_obj.get_fdata()

# load in a spinal cord mask
mask_obj = nib.load(mask_sc_fname)
mask = mask_obj.get_fdata()
mask = mask.astype('bool')

# mask dti maps
fa_mskd = fa[mask]
md_mskd = md[mask]
ad_mskd = ad[mask]
rd_mskd = rd[mask]

# create X matrix (feature matrix)
X = np.vstack((fa_mskd, md_mskd, ad_mskd, rd_mskd)).T

# apply scaler on the feature vectors
pickle_in = open(os.path.join(PATH,'scaler.pickle'),'rb')
scaler = pickle.load(pickle_in)
X_std = scaler.transform(X)


# perform segmentation using different model predictions

'''     SVM     '''
pickle_in = open(os.path.join(PATH,'clf_svm.pickle'),'rb')
svm = pickle.load(pickle_in)
y1 = svm.best_estimator_.predict(X_std)


'''     Logarithmic regression     '''
#pickle_in = open(os.path.join(PATH,'clf_logreg.pickle'),'rb')
#logreg = pickle.load(pickle_in)
#y2 = logreg.predict(X_std)



'''     remove small clusters     '''
mask1 = np.zeros(mask.shape)
mask1[mask] = y1

# find all your connected components (white blobs in your image)
mask1 = mask1.astype(np.uint8)*255
mask2 = np.zeros(mask1.shape)

# minimum size of particles we want to keep (number of pixels)
#here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
min_size = 20

for k in range(mask1.shape[2]):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask1[:,:,k], connectivity=8)
    #connectedComponentswithStats yields every seperated component with information on each of them, such as size
    #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    
    #your answer image
    tmp = np.zeros(output.shape)
    #for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            tmp[output == i + 1] = 255
    mask2[:,:,k] = tmp
        
        
'''   save segmentation as a binary nifti file   '''
nii_obj1 = nib.Nifti1Image(mask2, affine=mask_obj.affine, header=mask_obj.header)
nii_obj1.to_filename(os.path.join(PATH_test,'mask_gm_svm.nii'))