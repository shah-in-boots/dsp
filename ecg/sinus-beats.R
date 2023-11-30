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
# Filter out "unacceptable" beats 
# 	May need to be part of the segmentation feature
# 	Would null any beats or label them as "non-sinus"?
dat <- 
	beats[lengths(beats) != 0] |>
	lapply(extract_signal, data_format = 'matrix')

# How to PAD them to all be the same length
# Now each level of this is an ECG beat of the same length
# This could be slow though to mask it all
padded <- pad_sequences(dat, padding = 'post') 

# This needs to be masked later on (layer masking, or embedding with a zero mask)
# The length of the array is the number of files that will have some zero masking
masking_layer <- layer_masking()
unmasked_embedding <- tf$cast(
	tf$tile(tf$expand_dims(padded, axis = -1L), c(7L, 333L, 12L, 1L)), tf$float32
) 
masked_embedding <- masking_layer(unmasked_embedding) # This can be very slow

# We need to create a label system that matches this
# This means that the number of beats x the number of labels
n <- length(dat) 
named_labels <- c('blue', 'yellow', 'yellow', 'blue', 'yellow', 'blue', 'yellow')
labels <- c(0, 1, 1, 0, 1, 0, 1)

# Model definition

	#layer_input(batch_shape = c(7, 12)) |>
	#layer_masking(masked_embedding) |>
	#layer_flatten() |>
	#layer_normalization() |>
	#layer_dense(units = 128, activation = 'relu') |>
	#layer_dense(units = 2, activation = 'softmax')

model <- keras_model_sequential() 
sig <- layer_input(shape = c(333, 12), dtype = 'float32')
flat <- layer_flatten(input_shape = c(333, 12))
dense <- layer_dense(units = 64, activation = 'relu')

x <- flat(sig)
x <- dense(flat(sig))

y <-
	layer_input(shape = c(333, 12), dtype = 'float32') |>
	layer_embedding(input_dim = 7, output_dim = 2, mask_zero = TRUE) |>
	layer_flatten()
	
inputs <- layer_input(shape = c(333, 12), dtype = 'float32')
	
outputs <- 
	inputs |>
	#layer_embedding(input_dim = 12, output_dim = 2, mask_zero = TRUE) |>
	layer_masking() |>
	layer_flatten() |>
	layer_normalization() |>
	layer_dense(units = 128, activation = 'relu') |>
	layer_dense(units = 2, activation = 'softmax')
	
model <- keras_model(inputs = inputs, outputs = outputs, name = 'mdl')

# Model compilation

model |>
	compile(
		optimizer = 'adam',
		loss = loss_sparse_categorical_crossentropy(),
		metrics = 'accuracy'
	)

# Training
history <-
	model |> fit(padded, labels, epochs = 30, verbose = 2)
