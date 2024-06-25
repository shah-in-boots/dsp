# Libraries and data ----
library(reticulate)
library(tensorflow)
library(tfdatasets)
library(keras)
library(tidyverse)
library(tidymodels)
EGM:::set_wfdb_path("/usr/local/bin")
library(EGM)

# Training dataset
tbl <- targets::tar_read(ttn_train_dataset, store = "~/OneDrive - University of Illinois Chicago/targets/aflubber/")

# For class weights
shuffleInd <- sample(nrow(tbl))
xa <- array(unlist(tbl$x[shuffleInd, drop = FALSE]), dim = c(nrow(tbl), 500, 12, 1))
ya <- array(unlist(tbl$y[shuffleInd, drop = FALSE]), dim = nrow(tbl))


# Approach by Van De Leur ----

vmdl <-
  keras_model_sequential() |>
  # Block 1
  layer_conv_1d(filters = 128, kernel_size = 3, dilation_rate = 1, padding = "causal", input_shape = c(500, 12)) |>
  layer_batch_normalization() |>
  layer_activation_leaky_relu(alpha = 0.01) |>
  # Block 2
  layer_conv_1d(filters = 128, kernel_size = 3, dilation_rate = 2, padding = "causal") |>
  layer_batch_normalization() |>
  layer_activation_leaky_relu(alpha = 0.01) |>
  # Block 3
  layer_conv_1d(filters = 128, kernel_size = 3, dilation_rate = 4, padding = "causal") |>
  layer_batch_normalization() |>
  layer_activation_leaky_relu(alpha = 0.01) |>
  # Block 4
  layer_conv_1d(filters = 128, kernel_size = 3, dilation_rate = 8, padding = "causal") |>
  layer_batch_normalization() |>
  layer_activation_leaky_relu(alpha = 0.01) |>
  # Block 5
  layer_conv_1d(filters = 128, kernel_size = 3, dilation_rate = 16, padding = "causal") |>
  layer_batch_normalization() |>
  layer_activation_leaky_relu(alpha = 0.01) |>
  # Block 6
  layer_conv_1d(filters = 128, kernel_size = 3, dilation_rate = 32, padding = "causal") |>
  layer_batch_normalization() |>
  layer_activation_leaky_relu(alpha = 0.01) |>
  # Block 7
  layer_conv_1d(filters = 256, kernel_size = 3, dilation_rate = 64, padding = "causal") |>
  layer_batch_normalization() |>
  layer_activation_leaky_relu(alpha = 0.01) |>
  # Global max pooling
  layer_global_max_pooling_1d() |>
  # Dense layer for output
  layer_dense(units = 1, activation = "sigmoid")

# Compile the model
vmdl |>
  keras::compile(
    optimizer = optimizer_adam(learning_rate = 0.0001, amsgrad = FALSE),
    loss = keras::loss_binary_crossentropy(),
    metrics = list(
      keras::metric_binary_accuracy(name = "acc"),
      keras::metric_true_negatives(name = "tn"),
      keras::metric_true_positives(name = "tp"),
      keras::metric_false_negatives(name = "fn"),
      keras::metric_false_positives(name = "fp"),
      keras::metric_auc(name = "auc")
    )
  )

class_wt <- list(
  "0" = 1 / sum(tbl$y == 0),
  "1" = 1 / sum(tbl$y == 1)
)

history <-
  vmdl |>
  fit(
    x = xa,
    y = ya,
    validation_split = 0.2,
    epochs = 10,
    class_weight = class_wt,
    callbacks = list(
      callback_early_stopping(monitor = "val_loss", patience = 10),
      callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1, patience = 5)
    )
  )

# Test model fit
dat <- targets::tar_read(ttn_test_dataset, store = "~/OneDrive - University of Illinois Chicago/targets/aflubber/")
xt <- array(unlist(dat$x[shuffleInd, drop = FALSE]), dim = c(nrow(dat), 500, 12, 1))
yt <- array(unlist(dat$y[shuffleInd, drop = FALSE]), dim = nrow(dat))

predictions <-
  vmdl |>
  predict(xt)

# Grad-CAM ----

# Last convolutional layer
last_conv_layer <-
  sapply(vmdl$layers, function(.x) {
    .x$name
  }) |>
  grep("conv1d", x = _, value = TRUE) |>
	tail(n = 1L)

# Compute class activation

gradModel <-
	keras_model(
		inputs = vmdl$input, 
		outputs = list(vmdl$get_layer(last_conv_layer)$output,  vmdl$output)
	)

# Get beats of interest
cases <- tbl$x[tbl$y == 1]
beats <- array(unlist(cases), dim = c(length(cases), 500, 12, 1))
singleBeat <- beats[1, , , ]
singleInput <- array_reshape(singleBeat, c(1, 500, 12))  # Reshape it for the model

	
with(tf$GradientTape() %as% tape, {
	# Inside the tape context, all operations are recorded
	outputs <- gradModel(singleInput)
	conv_outputs <- outputs[[1]]
	predictions <- outputs[[2]]
	
	# Identify the class index for which to compute the gradient
	pred_index <- tf$argmax(predictions[1,])
	
	# Focus on the output value corresponding to that class
	class_channel <- predictions[, pred_index]
})

# Compute gradients
grads <- tape$gradient(class_channel, conv_outputs)
pooled_grads <- tf$reduce_mean(grads, axis = c(1L, 2L))
pooled_grads <- tf$reduce_mean(grads, axis = tf$constant(c(1, 2), dtype = tf$int32))

# Work on heatmap
#conv_outputs <- conv_outputs[1,]
heatmap <- conv_outputs * pooled_grads
heatmap <- tf$reduce_mean(heatmap, axis = -1L)

# Normalize heatmap
heatmap <- tf$maximum(heatmap, 0) / tf$reduce_max(heatmap)
heatmap <- as.array(heatmap)
heatmap <- array_reshape(heatmap, c(500, 1))

plot(heatmap, type = "l", main = "Grad-CAM Heatmap")





# Function to compute Grad-CAM
compute_gradcam <- function(model, img_array, last_conv_layer_name, pred_index = NULL) {
	grad_model <- keras_model(inputs = model$input, outputs = list(
		model$get_layer(last_conv_layer_name)$output,
		model$output
	))
	
	with(tf$GradientTape() %as% tape, {
		c(conv_outputs, predictions) <- grad_model(img_array)
		if (is.null(pred_index)) {
			pred_index <- tf$argmax(predictions[1,])
		}
		class_channel <- predictions[, pred_index]
	})
	
	grads <- tape$gradient(class_channel, conv_outputs)
	pooled_grads <- tf$reduce_mean(grads, axis = c(1, 2))
	
	conv_outputs <- conv_outputs[1,]
	heatmap <- conv_outputs * pooled_grads
	heatmap <- tf$reduce_mean(heatmap, axis = -1)
	
	heatmap <- tf$maximum(heatmap, 0) / tf$reduce_max(heatmap)
	heatmap
}

# Apply Grad-CAM
img_array <- array_reshape(as.matrix(new_data[1, , ]), c(1, 500, 12))
heatmap <- compute_gradcam(model, img_array, last_conv_layer_name)

# Plot the heatmap
heatmap <- as.array(heatmap)
heatmap <- array_reshape(heatmap, c(500, 1))
plot(heatmap, type = "l")
