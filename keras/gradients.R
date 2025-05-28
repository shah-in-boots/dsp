# This is hte file that I have most recently used to train the model
# Last run 2025-04-20

# Model training for gradients ----
library(reticulate)
library(tensorflow)
library(tfdatasets)
library(keras3)
library(tidyverse)
library(tidymodels)
library(abind)
EGM:::set_wfdb_path("/usr/local/bin")
library(EGM)

# Training dataset
tbl <- targets::tar_read(ttn_train_dataset, store = "~/OneDrive - University of Utah/targets/aflubber/")

# For class weights
tbl <- slice_sample(tbl, n = nrow(tbl))
shuffleInd <- sample(nrow(tbl))
xa <-
  tbl$x[shuffleInd, drop = FALSE] |>
  abind::abind(along = 3) |>
	aperm(perm = c(3, 1, 2)) 
	#array_reshape(x = _, dim = c(dim(xa)[1], dim(xa)[2], dim(xa)[3], 1))
ya <- tbl$y[shuffleInd, drop = FALSE] |> as.array()

# Model by Van De Leur
inputs <- layer_input(shape = c(500, 12, 1))
outputs <-
  inputs |> 
  # Block 1
  layer_conv_2d(filters = 128, kernel_size = c(3, 1), dilation_rate = c(1, 1), padding = "same") |>
  layer_batch_normalization() |>
  layer_activation_leaky_relu(negative_slope = 0.01) |>
  # Block 2
  layer_conv_2d(filters = 128, kernel_size = c(3, 1), dilation_rate = c(2, 1), padding = "same") |>
  layer_batch_normalization() |>
  layer_activation_leaky_relu(negative_slope = 0.01) |>
  # Block 3
  layer_conv_2d(filters = 128, kernel_size = c(3, 1), dilation_rate = c(4, 1), padding = "same") |>
  layer_batch_normalization() |>
  layer_activation_leaky_relu(negative_slope = 0.01) |>
  # Block 4
  layer_conv_2d(filters = 128, kernel_size = c(3, 1), dilation_rate = c(8, 1), padding = "same") |>
  layer_batch_normalization() |>
  layer_activation_leaky_relu(negative_slope = 0.01) |>
  # Block 5
  layer_conv_2d(filters = 128, kernel_size = c(3, 1), dilation_rate = c(16, 1), padding = "same") |>
  layer_batch_normalization() |>
  layer_activation_leaky_relu(negative_slope = 0.01) |>
  # Block 6
  layer_conv_2d(filters = 128, kernel_size = c(3, 1), dilation_rate = c(32, 1), padding = "same") |>
  layer_batch_normalization() |>
  layer_activation_leaky_relu(negative_slope = 0.01) |>
  # Block 7
  layer_conv_2d(filters = 128, kernel_size = c(3, 1), dilation_rate = c(64, 1), padding = "same") |>
  layer_batch_normalization() |>
  layer_activation_leaky_relu(negative_slope = 0.01) |>
  # Global max pooling
  layer_global_max_pooling_2d() |>
  # Dense layer for output
  layer_dense(units = 1, activation = "sigmoid")

vmdl <- keras_model(inputs = inputs, outputs = outputs)

# Compile the model
vmdl |>
  keras3::compile(
    optimizer = optimizer_adam(learning_rate = 0.0001), 
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
  "0" = 1,
  "1" = sum(ya == 0) / sum(ya == 1)
)

history <-
  vmdl |>
  fit(
    x = xa,
    y = ya,
    validation_split = 0.2,
    batch_size = 64,
    epochs = 10,
    class_weight = class_wt,
    callbacks = list(
      callback_early_stopping(monitor = "val_loss", patience = 5),
      callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1, patience = 5)
    )
  )

# Test model fit
dat <- targets::tar_read(ttn_test_dataset, store = "~/OneDrive - University of Utah/targets/aflubber/")
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
	caret::confusionMatrix(as.factor(y_pred), as.factor(yt), positive = "1")
print(confusion_matrix_details)

# Save model 
keras3::save_model(vmdl, "keras/ttn_model.keras", overwrite = TRUE)

# Gradient computation ----

vmdl <- keras3::load_model("keras/ttn_model.keras")
#vmdl <- keras3::load_model("keras/ttn_model_full.keras")

compute_gradients <- function(keras_mdl, layer_name, input_array) {
	
	# Get the specified layer - last convolutional layer
  selectedLayer <-
    sapply(keras_mdl$layers, function(.x) {
      .x$name
    }) |>
    grep(layer_name, x = _, value = TRUE) |>
    tail(n = 1L)
	
	# Create a model that outputs both the layer output and predictions
  gradModel <-
    keras_model(
      inputs = keras_mdl$input,
      outputs = list(keras_mdl$get_layer(selectedLayer)$output, keras_mdl$output)
    )
	
	with(tf$GradientTape() %as% tape, {
		outputs <- gradModel(input_array)
		convOutputs <- outputs[[1]]
		predictions <- outputs[[2]]
		classChannel <- predictions[,1]  # Assuming binary classification
	})
	
	# Compute gradients
	grads <- tape$gradient(classChannel, convOutputs)
	
	# Pool the gradients over time_steps and leads dimensions
	pooledGrads <- tf$reduce_mean(grads, axis = c(1L, 2L))  # Shape: [batch_size, filters]
	
	# Reshape pooledGrads to [batch_size, 1, 1, filters] for broadcasting
	pooledGrads <- tf$reshape(pooledGrads, shape = c(tf$shape(pooledGrads)[1], 1L, 1L, tf$shape(pooledGrads)[2]))
	
	# Multiply the feature maps by the pooled gradients
	weightedConvOutputs <- convOutputs * pooledGrads  # Broadcasting over time_steps and leads
	
	# Compute the heatmap by summing over the filters dimension
	# Then normalize
	# Shape: [batch_size, time_steps, leads]
	# Get maximum values to help with normalization
	# Avoid the */0 issue
	heatmap <- 
		tf$reduce_sum(weightedConvOutputs, axis = -1L) |>
		tf$nn$relu()
	
	max_val <- tf$reduce_max(heatmap, axis = c(1L, 2L), keepdims = TRUE)
	heatmap <- heatmap / (max_val + 1e-10)  # Add epsilon to avoid division by zero
	
	# Return the heatmap as an array
	# Shape: [batch_size, time_steps, leads]
	as.array(heatmap)
}

beats <-
	tbl$x[tbl$y == 1] |>
  abind::abind(along = 3) |>
  aperm(perm = c(3, 1, 2))

# General mean value per time point and column
# Apply mean gradient across all positive ECGs
grads <- 
	compute_gradients(vmdl, "conv2d", beats) |>
	apply(c(2, 3), mean) |>
	as.data.frame()

# Visualize gradients ----


# Use the cases and plot an average beat for each ECG lead
# Make an array of all positive cases and all negative cases
posArray <- 
  tbl$x[tbl$y == 1] |>
  abind::abind(along = 3) |>
	aperm(perm = c(3, 1, 2)) 

negArray <-
	tbl$x[tbl$y == 0] |>
  abind::abind(along = 3) |>
	aperm(perm = c(3, 1, 2))

# Create mean/median beats for both positive and negative
posMean <- apply(posArray, c(2, 3), mean)
negMean <- apply(negArray, c(2, 3), mean)
posSE <- apply(posArray, c(2, 3), sd)
negSE <- apply(negArray, c(2, 3), sd)

posMedian <- apply(posArray, c(2, 3), median)
negMedian <- apply(negArray, c(2, 3), median)
posIQR <- apply(posArray, c(2, 3), IQR)
negIQR <- apply(negArray, c(2, 3), IQR)

# Create data set that has information
meanBeat <- 
	bind_rows(case = rownames_to_column(as_tibble(posMean), var = "time"), 
						control = rownames_to_column(as_tibble(negMean), var = "time"),
						.id = "status") |>
	pivot_longer(cols = -c(time, status), names_to = "lead", values_to = "mean") 

medianBeat <-
	bind_rows(case = rownames_to_column(as_tibble(posMedian), var = "time"), 
						control = rownames_to_column(as_tibble(negMedian), var = "time"),
						.id = "status") |>
	pivot_longer(cols = -c(time, status), names_to = "lead", values_to = "median")

seBeat <-
	bind_rows(case = rownames_to_column(as_tibble(posSE), var = "time"), 
						control = rownames_to_column(as_tibble(negSE), var = "time"),
						.id = "status") |>
	pivot_longer(cols = -c(time, status), names_to = "lead", values_to = "se")

iqrBeat <-
	bind_rows(case = rownames_to_column(as_tibble(posIQR), var = "time"), 
						control = rownames_to_column(as_tibble(negIQR), var = "time"),
						.id = "status") |>
	pivot_longer(cols = -c(time, status), names_to = "lead", values_to = "iqr")

beatData <-
	full_join(meanBeat, seBeat, by = c("time", "status", "lead")) |>
	full_join(medianBeat, by = c("time", "status", "lead")) |>
	full_join(iqrBeat, by = c("time", "status", "lead")) |>
	mutate(time = as.numeric(time))

# Plot case and controls without confidence intervals
# Apply gradient data to the plot
names(grads) <- unique(beatData$lead)
gradData <-
	grads |>
	rownames_to_column(var = "time") |>
	mutate(time = as.numeric(time)) |>
	# Remove weights that are pre-P wave or post T wave
	mutate(across(I:V6, ~ signal::sgolayfilt(.x, n = 21))) |>
	# Apply SG filter for smoothing
	mutate(across(I:V6, ~ if_else(time < 120 | time > 420, 0, .x))) |>
	pivot_longer(cols = -time, names_to = "lead", values_to = "gradient") 

beatGradients <-
	beatData |>
	select(time, status, lead, median) |>
	pivot_wider(names_from = status, values_from = median) |>
	inner_join(gradData, by = c("time", "lead")) |>
	mutate(lead = factor(lead, levels = c("I", "AVR", "V1", "V4", "II", "AVL", "V2", "V5", "III", "AVF", "V3", "V6")))

ggplot(data = beatGradients) +
  # Add gradient background stripes
  geom_tile(aes(x = time, y = 0, fill = abs(gradient)), 
            alpha = 0.8,
            width = 1,
            height = Inf) +
  # Add case line in blue
  geom_line(aes(x = time, y = case, group = lead), 
            color = "darkgray", 
            size = 0.8) +
  # Add control line in red
  geom_line(aes(x = time, y = control, group = lead), 
            color = "skyblue4", 
            size = 0.8) +
  # Facet by lead
  facet_wrap(~lead, ncol = 4, scales = "free_y") +
  # Create a color scale matching the image
	scale_fill_gradient(
		low = "white",
		high = "indianred4",
	) + 
  # Customize theme and labels
  theme_void() +
	theme(
			legend.position = "none",
			panel.border = element_rect(fill = NA, colour = "black", linewidth = 1),
			strip.text = element_text(face = "bold")
		)

# Fill teh line by the gradient
ggplot(data = beatGradients, aes(x = time, y = case, group = lead)) +
	# Create a line for each lead where the color intensity varies by gradient
	geom_line(aes(y = case), color = "#000000", linewidth = 1.0) + 
	# Add control lines in a lighter color
	geom_line(aes(y = control), color = "#000000", linewidth = 1.0, alpha = 0.5) +
	geom_line(data = filter(beatGradients, gradient > 0.05),
            aes(x = time, y = case), 
            color = "#FEAB4D", 
            linewidth = 1) +
	# Facet by lead
	facet_wrap(~lead, ncol = 4) + 
	# Customize the color scale with limits to make small values more visible
	# Enhanced color scale with more contrast
	# Customize theme and labels
	theme_minimal() + 
	theme(
		strip.text = element_text(face = "bold"),
		panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.3),
    panel.spacing = unit(0, "lines"),
		legend.position = "none",
    axis.text = element_blank(),
    axis.ticks = element_blank(),
    axis.title = element_blank()
	) 



