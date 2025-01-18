# Install keras3
install.packages("keras3")
devtools::install_github("rstudio/keras")
keras3::install_keras(backend = "tensorflow")

# Getting started
library(keras3)
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

# Reshape
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))

# Rescale
x_train <- x_train / 255
x_test <- x_test / 255

# Labels
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

# Define model
model <- keras_model_sequential(input_shape = c(784))
model |>
	layer_dense(units = 256, activation = "relu") |>
	layer_dropout(rate = 0.4) |>
	layer_dense(units = 128, activation = "relu") |>
	layer_dropout(rate = 0.3) |>
	layer_dense(units = 10, activation = "softmax")

# Compile
model |>
	compile(
		loss = "categorical_crossentropy",
		optimizer = optimizer_rmsprop(),
		metrics = c("accuracy")
	)

# Train
history <- model |> fit(
	x_train, y_train,
	epochs = 30,
	batch_size = 128,
	validation_split = 0.2
)
