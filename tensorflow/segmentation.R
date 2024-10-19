# Libraries
library(keras)
library(tensorflow)

# Input layer
# 5000 steps and 1 channel at a time
input_shape <- c(5000, 1)
n_classes <- 4
inputs <- layer_input(shape = input_shape)

# Convolutional Block
conv_block <- 
	inputs |>
	layer_conv_1d(filters = 64, kernel_size = 5, activation = "relu", padding = "same") |>
	layer_batch_normalization() |>
	layer_max_pooling_1d(pool_size = 2)

# BiLSTM Block
bilstm_block <- 
	conv_block |>
	bidirectional(layer_lstm(units = 64, return_sequences = TRUE))

# Self Attention Block
attn_block <- 
	bilstm_block |>
	layer_dense(units = 1, activation = "tanh") |>
	layer_flatten() |>
	layer_activation("softmax") |>
	layer_repeat_vector(128) |>  
	layer_permute(c(2, 1))

mult_attn_block <- layer_multiply(list(bilstm_block, attn_block))

# Time Distributed Dense Layer
outputs <- 
	mult_attn_block |>
	time_distributed(layer_dense(units = n_classes, activation = "softmax"))

# Create and compile the model
cnn_bilstm_attn_model <- keras_model(inputs = inputs, outputs = outputs)

cnn_bilstm_attn_model |> compile(
	optimizer = optimizer_adam(),
	loss = "categorical_crossentropy",
	metrics = c("accuracy")
)
