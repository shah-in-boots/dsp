# Library setup ----
library(reticulate)
library(tensorflow)
library(tfdatasets)
library(keras)
library(tidyverse)
library(tidymodels)

# Beginner introduction ----

# Check to make sure its working
tf$constant("Hello Tensorflow!")

# Load a data set
dat <- keras::dataset_mnist()
x_train <- dat$train$x
y_train <- dat$train$y
x_test <- dat$test$x
y_test <- dat$test$y

# Normalize the X axis data (get it to be between 0 and 1)
x_train <- x_train / 255
x_test <- x_test / 255

# Build a sequential model of stacking layers
# The training data is 28 columns
model <-
	keras_model_sequential(input_shape = c(28, 28)) |>
	layer_flatten() |>
	layer_dense(units = 128, activation = "relu") |>
	layer_dropout(0.2) |>
	layer_dense(10)

predictions <- predict(model, x_train[1:2, , ])
predictions

# Uses the soft max predictions
tf$nn$softmax(predictions)

# Loss function...
loss_fn <-
	loss_sparse_categorical_crossentropy(from_logits = TRUE)

# Expect about loss of log(1/10 ~= 2.3) because there is about 1/10 error
loss_fn(y_train[1:2], predictions)

# Compile model
model |>
	compile(optimizer = 'adam',
					loss = loss_fn,
					metrics = 'accuracy')

# Training and fitting model
model |> fit(x_train, y_train, epoch = 5)
model |> evaluate(x_test, y_test, verbose = 2)

prob_model <-
	keras_model_sequential() |>
	model() |>
	layer_activation_softmax() |>
	layer_lambda(tf$argmax)

prob_model(x_test[1:5, , ])

# Advanced introduction ----

# This is above my level thus far

# Image classification ----

fashion_mnist <- keras::dataset_fashion_mnist()

c(train_images, train_labels) %<-% fashion_mnist$train
c(test_images, test_labels) %<-% fashion_mnist$test

# Images are 28x28 arrays (each spot has a pixel value)
# There are 60k training images and 10k test images
# The image data is thus N x 28 x 28, where images[1, , ] is 1st image values
dim(train_images)
dim(test_images)

# They are labeled from 0:9
dim(train_labels)
dim(test_labels)

# Labels are stored as numbers
class_names <- c(
	'T-shirt/top',
	'Trouser',
	'Pullover',
	'Dress',
	'Coat',
	'Sandal',
	'Shirt',
	'Sneaker',
	'Bag',
	'Ankle boot'
)

# Prerocessing the data
# Convert to wide format
# Give everyting X and Y axis
library(tidyr)
library(dplyr)
library(tibble)
library(ggplot2)
img1 <- as.data.frame(train_images[1, , ])
colnames(img1) <- seq_len(ncol(img1))
img1 <- 
	img1 |>
	rownames_to_column(var = "y") |>
	mutate(y = as.integer(y))

img1 <- 
	img1 |>
	pivot_longer(-y, names_to = "x", values_to = "value") |>
	mutate(x = as.integer(x))

# Picture of a shoe
ggplot(img1, aes(x = x, y = y, fill = value)) +
	geom_tile() +
	scale_fill_gradient(low = "white", high = "black", na.value = NA) +
	scale_y_reverse() +
	theme_minimal() + 
	theme(panel.grid = element_blank()) +
	theme(aspect.ratio = 1) + 
	xlab("") + 
	ylab("")

# However, for neural networks, have to get all nubmers "normalized"
train_images <- train_images / 255
test_images <- test_images / 255

# Display 25 images as an example, confirm its right
par(mfcol = c(5, 5))
par(mar = c(0, 0, 1.5, 0), xaxs = 'i', yaxs = 'i')
for (i in 1:25) {
	img <- train_images[i, , ]
	img <- t(apply(img, 2, rev))
	image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n',
				main = paste(class_names[train_labels[i] + 1]))
}

# Building the model
model <- 
	keras_model_sequential() |>
	# Transforms 2D -> 1D array = 784 px
	# Doesn't learn, only reformats
	layer_flatten(input_shape = c(28, 28)) |> 
	# Dense layer 1
	# Has 128 nodes/neurons
	# Uses relu activation layer (rectified linear unit)
	layer_dense(units = 128, activation = 'relu') |>
	# Has 10 nodes as a softmax layer
	# Returns an array of 10 probability scores that sum to 1
	# Each node has a score suggesting probability to one of hte 10 digit classes
	layer_dense(units = 10, activation = 'softmax')

# Compiling
model |>
	compile(
		optimizer = 'adam',
		loss = 'sparse_categorical_crossentropy',
		metrics = c('accuracy')
	)

# Training using training labels and training images
# Model will learn to associate the image with labels
# Subsequently will test it out
model |> 
	fit(train_images, train_labels, epochs = 5, verbose = 2)

# Accuracy assessment
score <-
	model |>
	evaluate(test_images, test_labels, verbose = 1)

cat("Test loss", score["loss"], "\n")
cat("Test accuracy", score["accuracy"], "\n")

predictions <-
	model |>
	predict(test_images)

# From all the predictions can see each one
# Can also evaluate a single image based on the model alone
# Should be the same for hte same image
predictions[1, ]

img <- test_images[1, , , drop = FALSE]
model |>
	predict(img)

class_names[which.max(predictions[1, ])]

# Test Classification ----

# Evaluate text classification from plain text
# Train on IMDB for sentiment analysis
library(tensorflow)
library(keras)
library(tfdatasets)

# Large file from IMDB movie reviews
# Will put this in gitignore...
url <- "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
dataset <- get_file(
	"aclImdb_v1",
	url,
	untar = TRUE,
	cache_dir = '.',
	cache_subdir = ''
)

# The dataset contains positive and negative text files of movie reviews
dataset_dir <- file.path("./tensorflow/aclImdb")
list.files(dataset_dir)
train_dir <- file.path(dataset_dir, 'train')
list.files(train_dir)

sample_file <- file.path(train_dir, 'pos/1181_9.txt')
readr::read_file(sample_file)

# Remove useless files and folders
# Then split the data... train, validate, test
remove_dir <- file.path(train_dir, 'unsup')
unlink(remove_dir, recursive = TRUE)

# Get text files from the directory structure
# There are 25k examples, and will use 80% for training
# Generating training data
batch_size <- 32
seed <- 42
raw_train_ds <- keras::text_dataset_from_directory(
	train_dir,
	batch_size = batch_size,
	validation_split = 0.2,
	subset = 'training',
	seed = seed
)

batch <- 
	raw_train_ds |>
	reticulate::as_iterator() |>
	coro::collect()

# Example of the review data
# Text contains raw text and occ HTML tags
# Class names are also present in raw train data
batch[[1]][[1]][1] # Review
batch[[1]][[2]][1] # Shape of object
cat("Label 0 corresponds to", raw_train_ds$class_names[1])
cat("Label 1 corresponds to", raw_train_ds$class_names[2])

# Create a validation set
raw_val_ds <- keras::text_dataset_from_directory(
	train_dir,
	batch_size = batch_size,
	validation_split = 0.2,
	subset = 'validation',
	seed = seed
)

# Create test data set
raw_test_ds <- keras::text_dataset_from_directory(
	train_dir,
	batch_size = 32
)

# Since the data is text, needs to be cleaned/prepared
# Need to remove punctuation or HTML elements
# Then tokenize strings into tokens using a function
# Subsequently vectorized for feeding into neural network
re <- reticulate::import('re')
punctuation <- c("!", "\\", "\"", "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/", ":", ";", "<", "=", ">", "?", "@", "[", "\\", "\\", "]", "^", "_", "`", "{", "|", "}", "~")

punctuation_group <- 
	punctuation |>
	sapply(re$escape) |> # Adds escape before character
	paste0(collapse = "") |> # Collapses them
	sprintf(fmt = "[%s]")

standardize_text <- function(x) {
	lowercase <- tf$strings$lower(x)
	stripped_html <- tf$strings$regex_replace(lowercase, '<br />', ' ')
	tf$strings$regex_replace(
		stripped_html,
		punctuation_group,
		""
	)
}

# Will need to create a `TextVectorization` layer...
# Each token will get an index value as an integer
max_features <- 1e4
sequence_length <- 250 # Length of characters
vectorize_layer <- keras::layer_text_vectorization(
	standardize = standardize_text,  # Our custom function to make tokens
	max_tokens = max_features,
	output_mode = "int",
	output_sequence_length = sequence_length
)

# Next need to `adapt()` data from preprocessing into dataset
# Converts everything from strings -> integers
train_text <- 
	raw_train_ds |>
	dataset_map(function(text, label) { text })

vectorize_layer |>
	adapt(train_text)

vectorize_text <- function(text, label) {
	text <- tf$expand_dims(text, -1L)
	list(vectorize_layer(text), label)
}

# We can check to see how this looks
# Get a batch of 32 reviews + labels and evaluate them as vectors
# Also can evaluate what an integer value corresponds too
batch <- 
	reticulate::as_iterator(raw_train_ds) |>
	reticulate::iter_next()

first_review <- as.array(batch[[1]][1])
first_label <- as.array(batch[[2]][1])
cat("Review:", first_review)
cat("Label:", raw_train_ds$class_names[first_label + 1])
cat("Vectorized review:\n")
print(vectorize_text(first_review, first_label))

cat("9257 -->", get_vocabulary(vectorize_layer)[9257 + 1])
cat("15 -->", get_vocabulary(vectorize_layer)[15 + 1])
cat("Vocabulary size:", length(get_vocabulary(vectorize_layer)))

# Now can apply text vectorization
train_ds <- raw_train_ds |> dataset_map(vectorize_text)
val_ds <- raw_val_ds |> dataset_map(vectorize_text)
test_ds <- raw_test_ds |> dataset_map(vectorize_text)

# Now, the file size is very large overall
# We can put data on a cache to help with speed
at <- tf$data$AUTOTUNE
train_ds <- train_ds |> dataset_cache() |> dataset_prefetch(buffer_size = at)
val_ds <- val_ds |> dataset_cache() |> dataset_prefetch(buffer_size = at)
test_ds <- test_ds |> dataset_cache() |> dataset_prefetch(buffer_size = at)


# Creating the model
model <-
	# Uses sequential modeling layers
	keras_model_sequential() |>
	# Then, creates first dense layer for embeddings
	# Takes integer-encoded reviews
	# Looks up embedding vector for each word index
	# These vectors are learned as model trains
	# Result dimensions = (batch, sequence, embedding0)
	layer_embedding(input_dim = max_features + 1, output_dim = 16) |>
	layer_dropout(0.2) |>
	# Global pool returns fixed length output vector
	# Averages over teh sequence dimensions
	# Allows model to handle variable length input
	layer_global_average_pooling_1d() |>
	# Piped down to 16 hidden unit dense layers
	layer_dropout(0.2) |>
	# Last layer is a single output node
	layer_dense(1)

# Loss function... binary since its a classificatin problem
model |>
	compile(
		loss = loss_binary_crossentropy(from_logits = TRUE),
		optimizer = 'adam',
		metrics = metric_binary_accuracy(threshold = 0)
	)

# Training the model over epochs
history <- 
	model |>
	fit(train_ds, validation_data = val_ds, epochs = 10)
plot(history)

# Evaluate model for performance by LOSS (error) and ACCURACY
model |> evaluate(test_ds)

# Exporting the model with the text vectorization layer
# For it to work with raw strings (instead of processed) can give it this fn
export_model <-
	keras_model_sequential() |>
	vectorize_layer() |>
	model() |>
	layer_activation(activation = 'sigmoid') |>
	compile(
		loss = loss_binary_crossentropy(from_logits = FALSE),
		optimizer = 'adam',
		metrics = 'accuracy'
	)

export_model |>
	evaluate(raw_test_ds)

examples <- c(
	"The movie was great!",
	"The movie was okay.",
	"The movie was terrible..."
)

predict(export_model, examples)

# Basic Regression ----

# Cars dataset
url <-
	"http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"

col_names <-
	c(
		"mpg",
		"cylinders",
		"displacement",
		"horsepower",
		"weight",
		"acceleration",
		"model_year",
		"origin",
		"car_name"
	)

raw_dataset <- read.table(
	url,
	header = TRUE,
	col.names = col_names,
	na.strings = "?"
)

dataset <- select(raw_dataset, -car_name)

## Data preparation ----

# Clean data, as has 6 rows with missing information
lapply(dataset, function(.x) sum(is.na(.x))) |> str()
dataset <- na.omit(dataset)

# Add one-hot encoding for the origin of the car
dataset <-
	recipe(mpg ~ ., dataset) |>
	step_num2factor(origin, levels = c("USA", "Europe", "Japan")) |>
	step_dummy(origin, one_hot = TRUE) |>
	prep() |>
	bake(new_data = NULL)
	
# Split data into training and test sets
split <- initial_split(dataset, 0.8)
train_dataset <- training(split)
test_dataset <- testing(split)

# Evaluate the overall patterns in the data
skimr::skim(train_dataset)
x <-
	train_dataset |>
	corrr::correlate() |>
	corrr::focus(-model_year, -origin, mirror = TRUE) |>
	corrr::rearrange() |>
	corrr::shave()

corrr::fashion(x)
corrr::rplot(x)

# Get label values off of data, will need this for training and testing
train_features <- select(train_dataset, -mpg)
test_features <- select(test_dataset, -mpg)

train_labels <- select(train_dataset, mpg)
test_labels <- select(test_dataset, mpg)

# Can use keras to normalize data in a layer
# Normalize data to help with training to avoid scaling issues
# Needs to preprocess the data
normalizer <- layer_normalization(axis = -1L)
normalizer |> adapt(as.matrix(train_features)) # Stores mean/variance in layer
print(normalizer$mean)
first <- as.matrix(train_features[1, ])
cat('First example:', first)
cat('Normalized:', as.matrix(normalizer(first)))

## Single linear regression ----

# Lets try to predict `mpg` from `horsepower`
# Requires two steps...
# 	Normalize the `horsepower` feature
# 	Apply linear transformation with `y = mx + b` to create one output (dense)
horsepower <- matrix(train_features$horsepower)
horsepower_normalizer <- layer_normalization(input_shape = shape(1), axis = NULL)
horsepower_normalizer |> adapt(horsepower)

# Next need to create keras model
horsepower_model <-
	keras_model_sequential() |>
	horsepower_normalizer() |>
	layer_dense(units = 1)
summary(horsepower_model)

# As a toy example, can run it on a tiny subset of data
predict(horsepower_model, horsepower[1:10, ])

# Compile and then train/fit
horsepower_model |>
	compile(
		optimizer = optimizer_adam(learning_rate = 0.1),
		loss = 'mean_absolute_error'
	)

history <-
	horsepower_model |>
	fit(
		as.matrix(train_features$horsepower),
		as.matrix(train_labels),
		epochs = 100,
		verbose = 0,
		validation_split = 0.2
	)

plot(history)

# Test results can be saved
test_results <- list()
test_results[['horsepower_model']] <-
	horsepower_model |>
	evaluate(as.matrix(test_features$horsepower),
					 as.matrix(test_labels),
					 verbose = 1)

# The model is a single linear variable
# Can visualize predictions based on a single input
x <- seq(0, 250, length.out = 251)
y <- predict(horsepower_model, x)
ggplot(train_dataset) +
	geom_point(aes(x = horsepower, y = mpg, color = 'data')) +
	geom_line(data = data.frame(x, y), aes(x = x, y = y, color = 'prediction'))

## Multiple linear regression ----

# It will produce 1 output based on the dense layer
linear_model <-
	keras_model_sequential() |>
	normalizer() |> # Defined earlier
	layer_dense(units = 1)

# Should output 10 rows and 1 column
predict(linear_model, as.matrix(train_features[1:10, ]))

# The kernal weights should have shape of [9, 1]
linear_model$layers[[2]]$kernel 

# Compile, fit, and save results
linear_model |>
	compile(
		optimizer = optimizer_adam(learning_rate = 0.1),
		loss = 'mean_absolute_error'
	)

history <-
	linear_model |>
	fit(
		as.matrix(train_features),
		as.matrix(train_labels),
		epochs = 50,
		verbose = 1,
		validation_split = 0.2
	)

plot(history)

test_results[['linear_model']] <- 
	linear_model |>
	evaluate(as.matrix(test_features),
					 as.matrix(test_labels),
					 verbose = 0)

## Deep neural network (DNN) ----

# Can also make single and multiple input DNN models
# Similar to linear models except there are now hidden layers
# 	Normalization layer
# 	Two hidden, non-linear DENSE layers using ReLU activation
#		Linear DENSE single output layer
# Hidden = not directly connected to input or outputs

build_and_compile_model <- function(norm) {
	model <-
		keras_model_sequential() |>
		norm() |>
		layer_dense(64, activation = 'relu') |>
		layer_dense(64, activation = 'relu') |>
		layer_dense(1)
	
	model |>
		compile(
			loss = 'mean_absolute_error',
			optimizer = optimizer_adam(0.001)
		)
	
	model
}

# Include previously made normalization layer
dnn_hp_model <- build_and_compile_model(horsepower_normalizer)

# Has many more parameters than ca be trained
summary(dnn_hp_model)
history <- 
	dnn_hp_model |>
	fit(
		as.matrix(train_features$horsepower),
		as.matrix(train_labels),
		validation_split = 0.2,
		verbose = 0,
		epochs = 100
	)

# Can see that it is quicker to train and less linear (better fit)
plot(history)
x <- seq(0, 250)
y <- predict(dnn_hp_model, x)
ggplot(train_dataset) + 
	geom_point(aes(x = horsepower, y = mpg, color = 'data')) +
	geom_line(data = data.frame(x, y), aes(x = x, y = y, color = 'prediction'))

test_results[['dnn_hp_model']] <- 
	dnn_hp_model |> 
	evaluate(
		as.matrix(test_features$horsepower),
		as.matrix(test_labels),
		verbose = 0
	)

# Also works for all features, not just horsepower
dnn_model <- build_and_compile_model(normalizer)
summary(dnn_model)
history <- 
	dnn_model |>
	fit(
		as.matrix(train_features),
		as.matrix(train_labels),
		validation_split = 0.2,
		verbose = 0,
		epochs = 100
	)

# Can see that it is quicker to train and less linear (better fit)
plot(history)

test_results[['dnn_model']] <- 
	dnn_model |> 
	evaluate(
		as.matrix(test_features),
		as.matrix(test_labels),
		verbose = 0
	)

## Performance ----

print(test_results)

# Predictions
test_predictions <- predict(dnn_model, as.matrix(test_features))
ggplot(data.frame(pred = as.numeric(test_predictions), mpg = test_labels$mpg)) +
	geom_point(aes(x = pred, y = mpg)) +
	geom_abline(intercept = 0, slope = 1, color = 'blue')

# Error distribution
error <- test_predictions - test_labels$mpg
ggplot(data = data.frame(error = error)) + 
	geom_density(aes(x = error))

# Save model...
save_model_tf(dnn_model, './tensorflow/dnn_model')
reloaded <- load_model_tf('./tensorflow/dnn_model')

test_results[['reloaded']] <-
	reloaded |>
	evaluate(as.matrix(test_features),
					 as.matrix(test_labels),
					 verbose = 0)