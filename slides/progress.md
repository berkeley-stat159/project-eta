% Project Eta Progress Report
% Jon Jara, Will Sanderson, Juan Shishido, Paul Wu, Wendy Xu
% November 12, 2015

# Introduction

## The Paper

The Neural Basis of Loss Aversion in Decision-Making Under Risk

Authors: Tom, Fox, Trepel, and Poldrak (2007)

Idea:

> We investigated neural correlates of loss aversion while individuals decided
whether to accept or reject gambles that offered a 50/50 change of gaining or
losing money.

## The Data

- from OpenFMRI.org
    - `ds005`
- 16 subjects
- 3 conditions per subject

## The Method

- iteratively weighted least squares
    - used to reduce outliers
- logistic regression
    - gain and loss

# Initial Work

## Basics

- downloaded data
    - created a `bash` script included in `Makefile`
- convolution
    - gamma for hemodynamic
    - non-constant repetition time for neural model
- regression
- RMS

## Example Plot

![Middle slice of $\hat{\beta}_1$](image/beta1_middle_slice.png)

# Plan

## Statistical Analyses

- linear regression
- multi-pattern voxel analysis

## Regression

- behavioral data
- neural data

## Multi-Pattern Voxel Analysis

Used for estimating the whole brain.

Norman, Polyn, Detre, and Haxby (2006)

## Tools

- NumPy
- Nibabel
- Statsmodels
- Scikit-Learn
- Nilearn
- PyMVPA
- Matplotlib
- Seaborn

# Process

## Most Difficult Part

- defining and scoping an analysis plan
    - data approaches
    - analyses methods
- Git workflow
    - coordination
