# Creating datasets in tensorflow in R
library(tensorflow)
library(tfdatasets)
library(keras)

# There are two ways to make datasets in tensorflow. (1) A data source
# constructs a `Dataset` from data stored or read in from memory (2) A data
# transformation constructs a dataset from >= 1 `Dataset` objects
#
# By definition, a `Dataset` is a sequence of elements, where each element is
# the SAME nested structure of components Individual components can be any
# tensorflow structure, such as sparse or ragged tensors


## Reading input data as arrays ----

# This splits the MNIST data into a train/test
# Each of these is a list of length two (x and y)
# Image density normalization by dividing by 255
c(train, test) %<-% dataset_fashion_mnist()
# Then can normalize the imaging data (stored in [[1]])
train[[1]][] <- train[[1]]/255
dataset <- tensor_slices_dataset(train)

## Reading TFRecord data ----

# Creates a dataset that reads all of the examples from two files.
fsns_test_file <- get_file(
	"fsns.tfrec", 
	"https://storage.googleapis.com/download.tensorflow.org/data/fsns-20160927/testdata/fsns-00000-of-00001"
)

dataset <- tfrecord_dataset(filenames = list(fsns_test_file))

## Reading CSV data ----

titanic_file <- get_file(
	"train.csv", 
	"https://storage.googleapis.com/tf-datasets/titanic/train.csv"
)

# Data has 627 "rows", and 10 "columns"
df <- readr::read_csv(titanic_file)

# If fits into memory, can use `tensor_slices_dataset` to convert to make slices
titanic_slices <- tensor_slices_dataset(df)

# Must add iterator method to look through slices
# This `str()` describes each "column" of data
titanic_slices |>
	reticulate::as_iterator() |>
	reticulate::iter_next() |>
	str()

# Can also read in data from CSV files PRN
titanic_batches <- tf$data$experimental$make_csv_dataset(
	titanic_file,
	batch_size = 4L,
	label_name = "survived"
)

titanic_batches |>
	reticulate::as_iterator() |>
	reticulate::iter_next() |>
	str()

## Reading in sets of files ----

# This is useful when datasets are distributed as a set of files
# Each file is a case example

# This is a root folder with multiple subfolders
flowers_root <- get_file(
	'flower_photos',
	'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
	untar = TRUE
)

# Each subfolder is the label feature for each example
# Each file/example is a JPG
fs::dir_ls(flowers_root)
list_ds <- file_list_dataset(fs::path(flowers_root, "*", "*"))
list_ds |>
	dataset_take(5) |>
	coro::collect() |>
	str()

# The data then can be read in, extracting the label from the path
process_path <- function(file_path) {
	label <- tf$strings$split(file_path, "/")[-2]
	list(
		tf$io$read_file(file_path),
		label
	)
}

labeled_ds <- dataset_map(list_ds, map_func = process_path)

# If we don't add iteration methods, cannot move through the objects
el <- 
	labeled_ds |>
	reticulate::as_iterator() |>
	reticulate::iter_next()
el[[1]]$shape
el[[2]]

bds <- labeled_ds |> dataset_batch(5)

bds |> 
	coro::collect(7) |>
	str()

## Processing epochs ----

# {tfdatasets} allows processing multiple epochs
# Easiest way is to use `dataset_repeat()` to transform dataset

titanic_file <- get_file(
	"train.csv", 
	"https://storage.googleapis.com/tf-datasets/titanic/train.csv"
)

titanic_lines <- text_line_dataset(titanic_file)

# Can then make a function to get batch sizes
plot_batch_sizes <- function(ds) {
	batch_sizes <- ds |>
		coro::collect() |>
		sapply(function(x) as.numeric(x$shape[1]))
	
	plot(seq_along(batch_sizes), batch_sizes)
}

# Using `dataset_repeat()` allows dataset to be repeated "indefinitely"
titanic_batches <-
	titanic_lines |>
	dataset_repeat(3) |>
	dataset_batch(128) 

titanic_batches |>
	coro::collect(1) 

plot_batch_sizes(titanic_batches)

# By changing order of batch and repeat, can get clear epoch separation
titanic_batches <-
	titanic_lines |>
	dataset_batch(128) |>
	dataset_repeat(3) 

plot_batch_sizes(titanic_batches)
