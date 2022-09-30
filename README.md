## Machine/Deep Learning - Neural Networks

Paweł Syty, Gdańsk University of Technology,
Institute of Physics and Computer Science

CompAS 2022 conference workshop 

### Working environment (Anaconda-based)

1. Install Anaconda and run Anaconda Prompt

2. Update conda (optional)
```bash
conda update -n base -c defaults conda
```

3. Create and activate environment
```bash
conda create --name ml-nn
conda activate ml-nn
```

4. Install `pip`
```bash
conda install pip
```

5. Install necessary packages using `pip` or `conda`
```bash
conda install matplotlib
pip install tensorflow
pip install tensorflow_addons
pip install scikit-learn
pip install imutils
pip install opencv-python
```

Preferred editor: `spyder`
```bash
pip install spyder
```

If you're not using Anaconda, just install Python 3.x, and all of the above packages using ```pip```.


### 1. Regression problem

Gold nanoparticles are covered by thin layer of aluminium oxide. 
Position of maximum (peak) of light absorption for such system, `M`, depends on 
parameters of that layer:

- `d` - layer thickness (nm)
- `A` - amplitude of the dielectric function (a.u.)
- `omega` - resonant frequency of the dielectric function (a.u.)

Basing on the set of available FDTD simulation results for several combinations of the above paramers, 
construct and train ML model, allowing for prediction for any set of these parameters
(within the reasonable range), e.g. prediction `(d, A, omega) -> M`


### 2. Classification problems

#### Numerical data

Nanoparticles made of gold+silver alloys are fabricated from gold and silver thin films (layers) 
of some thicknesses. Next, these films are annealed by some time in 550 Celsius.
In some cases the shape of the light absorption spectra (from UV-Vis experiment) is well defined 
(FWHM, Full Width at Half Maxiumum is easy to calculate), in some cases - not. 

Parameters of the samples:

- `Time` - annealing time (min)
- `Au` - gold layer thickness (nm)
- `Ag` - silver layer thickness (nm)

Basing on the set of available experimental data, construct and train ML model for prediction if for 
given set of parameters, FWHM will be present or not.
In the other words, classify the parameters of the sample `(Time, Au, Ag)` into the `Y` or `N` label.

#### Image data

SEM (Scanning Electron Microscope) and TEM (Transmission electron microscopy) images are given
(all images are labelled). Construct a model to classify any image (other than those in the training set) 
as `SEM` or `TEM` image.

Since training of the network on these SEM/TEM images might be time consuming (due to their sizes),
another, 'lighter' image dataset has been prepared: limited number of handwritten digits taken
from the MNIST dataset. Classify these images as the proper digit (`0`-`9`).

### 3. Clusterization

Let's assume, that we do not have labels for the above image classification problems 
(`SEM`/`TEM` or `0`...`9`).
As a consequence, training the network in the supervised way is not possible.
But we can stil use neural network (together with selected classification method)
to automatically divide these images into clusters of the similar properties.

The task: build so-called autoassociative neural network for extract features of the images,
and then use these features for classification by k-means method. Use SEM/TEM set, and (lighter)
MNIST set.
