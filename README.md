# image_processing
The requirements.txt can be used to get the correct things installed for the code to run 
utils.py is a little package I made with some of the universal processes (when we do the flow segmentation) e.g. sanity checker which overlays the segmentation on its slice 



Image_segmentation_3d:
1. Loads in the slices as a 3D image 
2. Crops to a cylinder 
3. Performs the non-local means filter 
4. Segments using the watershed segmentation 

This is for the dry scans. The values for the watershed segmentation are currently input by the user. However, we could automate this in the future. 

Flow segmentation processes the flow images. Still deciding the best method between differential between flow and wet scan, and then thresholding or if I should mask the pore space with the dry scan. The masking makes a large difference to connectivity, at little expense to saturation. This is a huge issue for the analysis we are considering
