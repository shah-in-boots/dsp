# Initial libraries for tensorflow
library(reticulate)
library(tensorflow)
library(keras)

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
