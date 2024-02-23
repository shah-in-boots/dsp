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


## Reading input data as arrays

# This splits the MNIST data into a train/test
# Each of these is a list of length two (x and y)
# Image density normalization by dividing by 255
c(train, test) %<-% dataset_fashion_mnist()
# Then can normalize the imaging data (stored in [[1]])
train[[1]][] <- train[[1]]/255
dataset <- tensor_slices_dataset(train)

## Reading TFRecord data

# Creates a dataset that reads all of the examples from two files.
fsns_test_file <- get_file(
	"fsns.tfrec", 
	"https://storage.googleapis.com/download.tensorflow.org/data/fsns-20160927/testdata/fsns-00000-of-00001"
)

dataset <- tfrecord_dataset(filenames = list(fsns_test_file))

## Reading in sets of files

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