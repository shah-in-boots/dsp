# Initial libraries for tensorflow
devtools::install_github("rstudio/tensorflow")

# Ensure python is available
reticulate::install_python()

# Create virtual environment? Not sure if needed
library(reticulate)
path_to_python <- Sys.getenv("RETICULATE_PYTHON")
virtualenv_create("r-reticulate", python = path_to_python)

library(tensorflow)
install_tensorflow(envname = "r-tensorflow")
tf$constant("Hello Tensorflow!")

devtools::install_github("rstudio/keras")
library(keras3)
install_keras()

# Check to make sure its working
reticulate::py_config()
tensorflow::tf_config()
keras3::config_backend()

