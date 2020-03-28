# PathFlow-MixMatch

Improved giga-pixel WSI registration through automated segment-based registration. Proof-of-concept pipeline, would love your contributions!     

Currently under review and additional validation, Biorxiv: https://www.biorxiv.org/content/10.1101/2020.03.22.002402v1  

UPDATE: We are in the process of updating this repository with the necessary level of documentation and adding a Wiki page, stay tuned!  

This package can be installed for Python 3.6+ using the following command:

```
pip install pathflow_mixmatch
```

Our latest package build can be installed using:

```
pip install git+https://github.com/jlevy44/PathFlow-MixMatch.git   
```

Minimal working example:  
```
pathflow-mixmatch --im1 A.png --im2 B.png --fix_rotation False --output_dir output_registered_images/ --gpu_device 0 --transform_type similarity --lr 0.01 --iterations 1000 --min_object_size 50000
```

See https://airlab.readthedocs.io/ for further description of available transformations and loss functions.  

Currently available loss functions:  
* mse  
* ncc  
* lcc  
* mi  
* mgf  
* ssim  

Currently available transformations:  
* similarity  
* affine  
* rigid  
