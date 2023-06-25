# Initial libraries for tensorflow
install.packages("tensorflow")
devtools::install_github("rstudio/keras")

# Create virtual environment
library(reticulate)
path_to_python <- Sys.getenv("RETICULATE_PYTHON")
virtualenv_create("r-reticulate", python = path_to_python)

library(tensorflow)
install_tensorflow(envname = "r-reticulate")
tf$constant("Hello Tensorflow!")

install.packages("keras")
library(keras)

# Check to make sure its working
reticulate::py_config()
tensorflow::tf_config()
