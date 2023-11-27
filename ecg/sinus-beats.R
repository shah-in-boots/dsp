# Libraries
library(reticulate)
library(tensorflow)
library(tfdatasets)
library(keras)
library(tidyverse)
library(tidymodels)
library(fs)
library(shiva)

# Eventually want to train a machine learning model based on this
# The goal is to create a clean training and testing dataset
# Training and test data:
# 	Train = milivolts on multiple leads of an ECG
# 	Test = labels for each dimension as a sequence (e.g. P wave on to off)

# Data location
ecg_dir <- 
	fs::path('~/OneDrive - University of Illinois Chicago/data/aflubber/ecg_data/wes/') |>
	fs::path_expand()
ecg_paths <-
	fs::dir_ls(ecg_dir, glob = '*.ecgpuwave') |>
	fs::path_ext_remove()

# Sample ECG to work with
dir <- fs::path_dir(ecg_paths[1])
rec <- fs::path_file(ecg_paths[1])
ecg <- shiva::read_wfdb(rec, dir, annotator = 'ecgpuwave')
ggm(ecg) |>
	draw_boundary_mask()
readr::read_lines(fs::path(ecg_paths[1], ext = 'hea'))
beats <- segmentation(ecg, by = 'sinus')

# Convert to a simple array
dat <- 
	beats[[1]]$signal[, -1] |>
	as.matrix()

array(dat) |> dim()

