## Physics-Informed Neural Operator (PINO) for Atrial Fibrillaion

This repo borrows from https://github.com/martavarela/EP-PINNs and https://github.com/neuraloperator/physics_informed to build a PINO for inverse parameter estimation for the AP model. Other useful repos are https://github.com/annien094/EP-PINNs-for-drugs and https://github.com/annien094/2D-3D-EP-PINNs.

The AP model is a set of PDEs that describes wave propagation over the surface of the atria. EP-PINNs have been shown to be accurate forward solvers of the AP model and accurate inverse parameter estimators. However, PINNs have some drawbacks that are addressed by PINOs. This project aims to train and test a PINO for inverse parameter estimation for use in EP research.

See the directory matlab_code for the script used to generate test data using a FD solver of the AP model. Due to github's file size constraints, we are unable to upload the generated data, but they can be downloaded here: https://drive.google.com/drive/folders/1QV9FRPbUbnCg8vLtgev91gX9AUqrDbE9?usp=sharing. 

The train_test_pino_inv.py file contains the full script, but pino_inv.ipynb can be used as well. All required packages can be found under requirements.txt.