# Image Processing Workflow
Python notebook that uses skimage packages to segment an image. Can be used as a subsitute to Avizo. 
The paper associated with this work is: https://doi.org/10.31223/X58D69

## Guide to getting the code running 
Recommended steps for running the notebook.

1. Download and install vscode: https://code.visualstudio.com/
2. Download and install python: https://www.python.org/downloads/ (download version 3.10.11)
3. Install the python extension in vscode. In vscode, click the extensions icon in the sidebar and search for "python"
4. Open a new folder (link it the folder containing the contents of this repository, wherever that is on your computer)
5. Create a virtual enviornment. In vscode click View>Terminal and enter the code below: <br> 
   python -m venv myenv <br> 
   Then for Mac enter: source myenv/bin/activate <br> 
   or for Windows: myenv\Scripts\activate 
7. Install the necessary packages by typing this in the terminal:
   pip install -r requirements.txt
   NB must be in the folder where the requirements.txt file is. Alternatively, you can install the packages individually e.g. pip install numpy.
   Now open the notebook (wavelet_plotting_GRL_paper.ipynb) and begin!

## Pore space segmentation 
This is done with image_segmentation_3D.ipynb. Here we segement the pore space using watershed segmentation. 

The pore space is loaded and cropped to remove everything that isn't the core (i.e. no sleeve or core holder in the images):
![image](https://github.com/cspurin/image_processing/assets/108369280/700c0e83-9ce7-4ecc-b45e-cf232a63d3d4)

The data is filtered using a non-local means filter:
![image](https://github.com/cspurin/image_processing/assets/108369280/04f43a9d-23c7-44e9-9739-4573e11c56d5)

Then the image is segmented using a watershed segmentation. Values for this are chosen by the user: 
![image](https://github.com/cspurin/image_processing/assets/108369280/da1dd766-0fe5-411d-8243-25162564bc78)


## Flow segmentation 
This is done with flow_segmentation_3D.ipynb. Here we segment the gas in the images using differential imaging and a simple threshold. This requires an image with only brine in the pore space prior to the injeciton of the gas (or oil). 

The wet image is cropped and registered to the dry scan.  
The flow image is cropped and registered to the dry scan. 

The images are filtered using the non-local means filter: 
![image](https://github.com/cspurin/image_processing/assets/108369280/84e58ae1-e8bf-4611-8890-95b39c8fa3b6)

We subtract the wet image from the flow image to get the location of the gas: 
![image](https://github.com/cspurin/image_processing/assets/108369280/6f9bbce0-92fb-4658-8f74-a95356929bea)

The segmented dry scan can get the location of the water in the final segmentation of the images. 

