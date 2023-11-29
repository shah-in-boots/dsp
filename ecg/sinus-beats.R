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
	beats[lengths(beats) != 0] |>
	lapply(extract_signal, data_format = 'matrix')

# How to PAD them to all be the same length
# Now each level of this is an ECG beat of the same length
# This also requires added 
padded <- pad_sequences(dat, padding = 'post', maxlen = max(lengths(dat))) # This doesn't work right, because it uses the matrix as a flattened vector first
padded <- pad_sequences(dat, padding = 'post') 

# This needs to be masked later on (layer masking, or embedding with a zero mask)
masking_layer <- layer_masking()
unmasked_embedding <- tf$cast(
	tf$tile(tf$expand_dims(padded, axis = -1L), c(7L, 333L, 12L, 1L)), tf$float32
) 
masked_embedding <- masking_layer(unmasked_embedding) # This can be very slow



# We need to create a label system that matches this
# This means that the number of beats x the number of labels
