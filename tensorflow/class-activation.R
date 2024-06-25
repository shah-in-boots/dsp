# Libraries and data ----
library(reticulate)
library(tensorflow)
library(tfdatasets)
library(keras)
library(tidyverse)
library(tidymodels)
library(abind)
EGM:::set_wfdb_path("/usr/local/bin")
library(EGM)

# Training dataset
tbl <- targets::tar_read(ttn_train_dataset, store = "~/OneDrive - University of Illinois Chicago/targets/aflubber/")

# For class weights
shuffleInd <- sample(nrow(tbl))
xa <-
  tbl$x[shuffleInd, drop = FALSE] |>
  abind::abind(along = 3) |>
  aperm(perm = c(3, 1, 2))
ya <- tbl$y[shuffleInd, drop = FALSE] |> as.array()


# Model by Van De Leur ----

vmdl <-
  keras_model_sequential() |>
  layer_masking() |>
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
  # LSTM
  layer_lstm(units = 64, return_sequences = TRUE) |>
  # Global max pooling
  layer_global_max_pooling_1d() |>
  # Dense layer for output
  layer_dense(units = 1, activation = "sigmoid")

# Compile the model
vmdl |>
  keras::compile(
    optimizer = optimizer_adam(learning_rate = 0.0001),
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
  "0" = 1 / sum(ya == 0),
  "1" = 1 / sum(ya == 1)
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
xt <-
  dat$x |>
  abind::abind(along = 3) |>
  aperm(perm = c(3, 1, 2))
yt <- dat$y |> as.array()

predictions <-
  vmdl |>
  predict(xt)

y_pred <- ifelse(predictions > 0.5, 1, 0)

# Optionally, you can use the `confusionMatrix` function from the `caret` package for more details
confusion_matrix_details <- 
	caret::confusionMatrix(as.factor(y_pred), as.factor(yt))
print(confusion_matrix_details)

# Grad-CAM ----

# Get beats of interest
cases <- tbl$x[tbl$y == 1]

beats <-
  abind::abind(cases, along = 3) |>
  aperm(perm = c(3, 1, 2))
n <- 10 # Beat number
singleBeat <- array(beats[n, , ], dim = c(1, 500, 12))

# Last convolutional layer
last_conv_layer <-
  sapply(vmdl$layers, function(.x) {
    .x$name
  }) |>
  grep("conv1d", x = _, value = TRUE) |>
  tail(n = 1L)

# Compute class activation

# First create a gradient model with output for last conv_1d layer as well
gradModel <-
  keras_model(
    inputs = vmdl$input,
    outputs = list(vmdl$get_layer(last_conv_layer)$output, vmdl$output)
  )


with(tf$GradientTape() %as% tape, {
  # Inside the tape context, all operations are recorded
  outputs <- gradModel(singleBeat)
  conv_outputs <- outputs[[1]]
  predictions <- outputs[[2]]

  # Identify the class index for which to compute the gradient
  pred_index <- tf$argmax(predictions[1, ])

  # Focus on the output value corresponding to that class
  class_channel <- predictions[, pred_index]
})

# Compute gradients
grads <- tape$gradient(class_channel, conv_outputs)
pooled_grads <- tf$reduce_mean(grads, axis = c(1L))

# Weight the feature maps
conv_shape <- conv_outputs[1, , ] # Shape: [500, 256]

# Reshape pooled_grads to match the feature map shape
pooled_grad_shape <- tf$reshape(pooled_grads, shape = c(1L, 256L))

heatmap <- conv_shape * pooled_grad_shape # Shape: [500, 256]
reduced_heatmap <- tf$reduce_sum(heatmap, axis = -1L) # Shape: [500]

# Normalize heatmap
normalized_heatmap <-
  tf$maximum(reduced_heatmap, 0) / tf$reduce_max(reduced_heatmap)
heatmap_array <- as.array(normalized_heatmap)

# Expand the heatmap to match the number of leads
expanded_heatmap <- matrix(heatmap_array, nrow = 500, ncol = 12, byrow = FALSE)

# Plot heatmap for a specific lead
plot(expanded_heatmap[, 1], type = "l", main = "Grad-CAM Heatmap for All Leads")

# Overlay onto original data
# Assuming singleBeat is your original input with shape [1, 500, 12]
original_input <- singleBeat[1, , ]

ggplot(as.data.frame(original_input)) +
  geom_line(aes(x = 1:500, y = V1)) +
  geom_line(aes(x = 1:500, y = V2)) +
  geom_line(aes(x = 1:500, y = V3)) +
  geom_line(aes(x = 1:500, y = V4)) +
  geom_line(aes(x = 1:500, y = V5)) +
  geom_line(aes(x = 1:500, y = V6)) +
  geom_line(aes(x = 1:500, y = V7)) +
  geom_line(aes(x = 1:500, y = V8)) +
  geom_line(aes(x = 1:500, y = V9)) +
  geom_line(aes(x = 1:500, y = V10)) +
  geom_line(aes(x = 1:500, y = V11)) +
  geom_line(aes(x = 1:500, y = V12)) +
  geom_vline(aes(xintercept = 1:500, alpha = expanded_heatmap[, 1]), color = "red") +
  scale_alpha_identity() +
  theme_minimal()

# Functional Grad-CAM ----

beats <-
	tbl$x[tbl$y == 1] |>
  abind::abind(along = 3) |>
  aperm(perm = c(3, 1, 2))

compute_gradients <- function(keras_mdl, layer_name, input_array) {
  # Last convolutional layer
  selectedLayer <-
    sapply(keras_mdl$layers, function(.x) {
      .x$name
    }) |>
    grep(layer_name, x = _, value = TRUE) |>
    tail(n = 1L)

  # Create a gradient keras_model with output for last conv_1d layer as well
  gradModel <-
    keras_model(
      inputs = keras_mdl$input,
      outputs = list(keras_mdl$get_layer(selectedLayer)$output, keras_mdl$output)
    )

  # Get new predictions
  with(tf$GradientTape() %as% tape, {
    # Inside the tape context, all operations are recorded
    outputs <- gradModel(input_array)
    convOutputs <- outputs[[1]]
    predictions <- outputs[[2]]

    # Identify the class index for which to compute the gradient
    predIndex <- tf$argmax(predictions[1, ])
    classChannel <- predictions[, predIndex]
  })

  # Compute gradients
  grads <- tape$gradient(classChannel, convOutputs)
  pooledGrads <- tf$reduce_mean(grads, axis = list(1L))
  reshapedPool <- tf$reshape(pooledGrads, shape = c(dim(pooledGrads)[1], 1L, dim(pooledGrads)[2]))

  # Weight the feature maps
  weightedConvOutput <- convOutputs * reshapedPool # Shape: [n, 500, 256]

  # Reduce along the channel dimension to get the heatmap
  # Normalize the heatmap
  heatmap <- tf$reduce_mean(weightedConvOutput, axis = -1L) # Shape: [n, 500]
  heatmap <-
    tf$maximum(heatmap, 0) / tf$reduce_max(heatmap, axis = 1L, keepdims = TRUE)
  heatmap <- as.array(heatmap)

  # Return heatmap
}

# Test this out on input array data
grads <- compute_gradients(vmdl, "conv1d", beats) |> as.data.frame()

# General mean/median value
meanGrads <- apply(grads, 2, mean)
meanGrads <- colSums(grads) / colSums(!!grads)
meanGrads[is.na(meanGrads)] <- 0

# Use the cases and plot an average beat for each ECG lead
ecg_array <-
  tbl$x[tbl$y == 1] |>
  abind::abind(along = 3)

# Calculate the median value for each time point and each lead
median_matrix <- apply(ecg_array, c(1, 2), median)

# Convert to long format
median_df <- as.data.frame(median_matrix)
median_df$Time <- 1:nrow(median_df)
median_df$Gradient <- meanGrads
median_long <- pivot_longer(median_df, cols = -c(Time, Gradient), names_to = "Lead", values_to = "Amplitude")
median_long$Gradient[median_long$Amplitude == 0] <- 0

# Plot using ggplot2
ggplot(median_long, aes(x = Time, y = Amplitude, color = Lead)) +
  # facet_wrap(~Lead) +
  geom_vline(aes(xintercept = Time, alpha = Gradient), linewidth = 2, color = "indianred") +
  geom_line(linewidth = 1.1) +
  labs(title = "Median Beat for Each Lead at Each Time Point", x = "Time", y = "Amplitude") +
  scale_color_viridis_d(option = "mako") +
  scale_alpha_identity() +
  theme_void()
