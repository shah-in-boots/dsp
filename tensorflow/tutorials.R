# Library setup ----
library(reticulate)
library(tensorflow)
library(tfdatasets)
library(keras)

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
