# Libraries and data ----
library(reticulate)
library(tensorflow)
library(tfdatasets)
library(keras)
library(tidyverse)
library(tidymodels)
shiva:::set_wfdb_path('/usr/local/bin')
library(shiva)

# Training dataset
dat <- targets::tar_read(ttn_train_dataset, store = "~/OneDrive - University of Illinois Chicago/targets/aflubber/")

ds <- 
	tensor_slices_dataset(dat) |>
	dataset_map(unname) |>
	dataset_shuffle(5000) |>
	dataset_batch(32)

# Modeling approach ---- 

# To create a CAM, network architecture needs to have the following
# 	Final convolutional layer
# 	Global average pooling layer
# 	Linear/dense layer

# Convolutional layer will change size based on number of filters
# 	If original size is 500 x 12 channels

gap <- keras::layer_global_average_pooling_1d(data_format = "channels_last")

# Convolutional model instead
mdl <- 
	keras_model_sequential(name = "sandbox", input_shape = c(500, 12)) |>
	layer_masking(mask_value = 0) |>
	layer_normalization() |>
	layer_conv_1d(
		filters = 32,
		kernel_size = 64,
	) |> 
	layer_conv_1d(
		filters = 32,
		kernel_size = 64,
	) |> 
	layer_dropout(0.5) |>
	layer_max_pooling_1d() |>
	layer_lstm(units = 128) |>
	layer_activation_leaky_relu() |>
	layer_dense(
		units = 64,
		kernel_initializer = "uniform"
	) |>
	layer_activation_leaky_relu() |>
	# Final layer
	layer_dense(
		units = 1,
		activation = "sigmoid"
	) 

mdl |> keras::compile(
	optimizer = "adam",
	loss = keras::loss_binary_crossentropy(),
	metrics = list(
		keras::metric_binary_accuracy(name = "acc"),
		keras::metric_false_negatives(name = "fn"),
		keras::metric_false_positives(name = "fp"),
		keras::metric_true_negatives(name = "tn"),
		keras::metric_true_positives(name = "tp"),
		keras::metric_auc(name = "auc")
	)
)

class_wt = list(
	"0" = 1 / sum(tbl$y == 0),
	"1" = 1 / sum(tbl$y == 1)
)

history <- 
	mdl |>
	fit(
		ds,
		epochs = 10,
		class_weight = class_wt
	)


history <- 
	mdl |>
	fit(
		xa,
		ya,
		epochs = 10,
		batch_size = 32,
		class_weight = class_wt
	)
