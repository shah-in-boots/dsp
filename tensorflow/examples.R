library(tensorflow)
library(keras)

# Timeseries classification ----

url <- "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA"

train_df <-
	get_file(
		fname = "FordA_TRAIN.tsv",
		origin = file.path(url, "FordA_TRAIN.tsv")
		) |>
	readr::read_tsv(col_names = FALSE)

train_df <- "FordA_TRAIN.tsv" %>%
	get_file(., file.path(url, .)) %>%
	readr::read_tsv(col_names = FALSE)
x_train <- as.matrix(train_df[, -1])
y_train <- as.matrix(train_df[, 1])

test_df <- "FordA_TEST.tsv" %>%
	get_file(., file.path(url, .)) %>%
	readr::read_tsv(col_names = FALSE)
x_test <- as.matrix(test_df[, -1])
y_test <- as.matrix(test_df[, 1])