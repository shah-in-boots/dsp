# Libraries and data ----
library(reticulate)
library(tensorflow)
library(tfdatasets)
library(keras3)
library(tidyverse)
library(tidymodels)
library(abind)
library(collapse)
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
  keras_model_sequential(input_shape = c(500, 12)) |>
  # Block 1
  layer_conv_1d(filters = 128, kernel_size = 3, dilation_rate = 1, padding = "causal") |>
  layer_batch_normalization() |>
  layer_activation_leaky_relu(alpha = 0.01) |>
  layer_dropout(rate = 0.2) |>
  # # Block 2
  # layer_conv_1d(filters = 128, kernel_size = 3, dilation_rate = 2, padding = "causal") |>
  # layer_batch_normalization() |>
  # layer_activation_leaky_relu(alpha = 0.01) |>
  # layer_dropout(rate = 0.2) |>
  # # Block 3
  # layer_conv_1d(filters = 128, kernel_size = 3, dilation_rate = 4, padding = "causal") |>
  # layer_batch_normalization() |>
  # layer_activation_leaky_relu(alpha = 0.01) |>
  # layer_dropout(rate = 0.2) |>
  # # Block 4
  # layer_conv_1d(filters = 128, kernel_size = 3, dilation_rate = 8, padding = "causal") |>
  # layer_batch_normalization() |>
  # layer_activation_leaky_relu(alpha = 0.01) |>
  # layer_dropout(rate = 0.2) |>
  # # Block 5
  # layer_conv_1d(filters = 128, kernel_size = 3, dilation_rate = 16, padding = "causal") |>
  # layer_batch_normalization() |>
  # layer_activation_leaky_relu(alpha = 0.01) |>
  # layer_dropout(rate = 0.2) |>
  # # Block 6
  # layer_conv_1d(filters = 128, kernel_size = 3, dilation_rate = 32, padding = "causal") |>
  # layer_batch_normalization() |>
  # layer_activation_leaky_relu(alpha = 0.01) |>
  # layer_dropout(rate = 0.2) |>
  # Block 7
  layer_conv_1d(filters = 256, kernel_size = 3, dilation_rate = 64, padding = "causal") |>
  layer_batch_normalization() |>
  layer_activation_leaky_relu(alpha = 0.01) |>
  layer_dropout(rate = 0.2) |>
  # Global max pooling
  layer_global_max_pooling_1d() |>
  # Dense layer for output
  layer_dense(
    units = 1, 
    activation = "sigmoid", 
    kernel_regularizer = regularizer_l1_l2(l1 = 0.01, l2 = 0.01)
  )



# Compile the model
vmdl |>
  keras3::compile(
    optimizer = optimizer_adam(learning_rate = 0.001),
    loss = keras3::loss_binary_crossentropy(),
    metrics = list(
      keras3::metric_binary_accuracy(name = "acc"),
      keras3::metric_true_negatives(name = "tn"),
      keras3::metric_true_positives(name = "tp"),
      keras3::metric_false_negatives(name = "fn"),
      keras3::metric_false_positives(name = "fp"),
      keras3::metric_auc(name = "auc")
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
    epochs = 1,
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

# Updated for keras3
gradModel <-
  keras_model(
    inputs = vmdl$inputs[[1]],
    outputs = list(vmdl$get_layer(last_conv_layer)$output, vmdl$outputs[[1]])
  )

# Gradient tape approach
with(tf$GradientTape() %as% tape, {
  # Inside the tape context, all operations are recorded
  outputs <- gradModel(singleBeat)
  conv_outputs <- outputs[[1]]
  predictions <- outputs[[2]]

  # Identify the class index for which to compute the gradient
  class_idx <- tf$argmax(predictions[1, ])

  # Focus on the output value corresponding to that class
  class_channel <- predictions[, class_idx]
})

# Compute gradients
grads <- tape$gradient(class_channel, conv_outputs)

# Manual gradients
gradModel <-
  keras_model(
    inputs = vmdl$inputs[[1]],
    outputs = list(vmdl$get_layer(last_conv_layer)$output, vmdl$outputs[[1]])
  )

input_tensor <- tf$constant(singleBeat, dtype = tf$float32)

outputs <- gradModel(input_tensor)
conv_outputs <- outputs[[1]]
predictions <- outputs[[2]]

loss_conv_outputs <- tf$reduce_sum(conv_outputs)
loss_predictions <- tf$reduce_sum(predictions)

# Manual gradients
grads <- lapply(gradModel$trainable_variables, function(.x) {
  tf$zeros_like(.x)
})

for (i in seq_along(gradModel$trainable_variables)) {
  var <- gradModel$trainable_variables[[i]]
  with(tf$GradientTape() %as% tape, {
    tape$watch(var)
    prediction <- gradModel(input_tensor)
    loss <- tf$reduce_sum(prediction)
  })
  grads[[i]] <- tape$gradient(loss, var)
}

grads <- tf$GradientTape(loss_predictions, gradModel$trainable_variables)


# Pooling
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
      inputs = keras_mdl$inputs[[1]],
      outputs = list(keras_mdl$get_layer(selectedLayer)$output, keras_mdl$outputs[[1]])
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

# Plotting ----

# General mean/median value
meanGrads <- apply(grads, 2, mean, na.rm = TRUE)
meanGrads <- colSums(grads, na.rm = TRUE) / colSums(!!grads, na.rm = TRUE)
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
  #facet_wrap(~Lead) +
  geom_vline(aes(xintercept = Time, alpha = Gradient), linewidth = 2, color = "indianred") +
  geom_line(linewidth = 1.0) + 
  labs(title = "Median Beat for Each Lead at Each Time Point", x = "Time", y = "Amplitude") +
  scale_color_viridis_d(option = "mako") +
  scale_alpha_identity() +
  theme_void()

# Plotting in van de leur style ----

# Simplify data first by random sampling
cases <- tbl[tbl$y == 1, ] |> slice_sample(n = 20)
controls <-
  tbl[tbl$y == 0, ] |> slice_sample(n = nrow(cases))
dat <- bind_rows(cases, controls) 

# Get lead names
leadNames <- dat$x[[1]] |> names()

# Individual data for plotting
individual_dat <-
  dat |>
  mutate(id = row_number()) |>
  mutate(status = factor(y)) |>
  select(-y) |>
  rowwise() |>
  mutate(signal = list(as.data.frame(x))) |>
  select(-x) |>
  unnest(signal) |>
  group_by(id) |>
  mutate(time = row_number()) |>
  pivot_longer(cols = -c(id, status, time), names_to = "lead", values_to = "amplitude") |>
  mutate(lead = factor(lead, levels = c("I", "AVR", "V1", "V4", "II", "AVL", "V2", "V5", "III", "AVF", "V3", "V6")))

# Median data
# Also where the gradient data should be included
grads <- compute_gradients(vmdl, "conv1d", beats) |> as.data.frame()
mean_gradients <- 
  grads |>
  sapply(mean) |>
  unname() |>
  tibble(gradient = _) |>
  rownames_to_column("time") |>
  # Mutae the low yield values
  mutate(gradient = if_else(gradient <= 0.05, 0, gradient))

median_dat <-
  individual_dat |>
  group_by(status, lead, time) |>
  collapse::fsummarise(amplitude = collapse::fmedian(amplitude)) |>
  collapse::join(x = _, y = mean_gradients, on = "time") 


# Plot all elements
ggplot(individual_dat, aes(x = time, y = amplitude, color = status)) +
  facet_wrap(~lead, nrow = 3, ncol = 4) + 
  #facet_wrap(~lead, ncol = 1, strip.position = "left") + 
  geom_vline(data = median_dat, aes(xintercept = time, alpha = gradient), color = "#ffee99", linewidth = 0.3) +
  geom_line(alpha = 0.01, linewidth = 1) +
  geom_line(data = median_dat, aes(y = amplitude), linewidth = 1.2) + 
  coord_cartesian(xlim = c(50, 450), ylim = c(-1200, 1200)) + 
  scale_color_manual(values = c("0" = "#2166ac", "1" = "#b2182b")) +
  scale_linewidth() + 
  #scale_alpha_identity() + 
  scale_alpha_continuous(range = c(0.1, 0.9)) + 
  theme_void() + 
  theme(legend.position = "bottom",
        panel.border = element_rect(color = "black", fill = NA, size = 1.5)) +
  labs(title = "Association of ECG with TTN Mutants with AF",
       caption = "Gradient values are highlighted in the background in yellow")
  
  
