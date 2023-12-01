# Libraries
library(reticulate)
library(tensorflow)
library(tfdatasets)
library(keras)
library(tidyverse)
library(tidymodels)
library(shiva)
library(fs)


# Data location
ecg_dir <- 
	fs::path('~/OneDrive - University of Illinois Chicago/data/aflubber/ecg_data/wes/') |>
	fs::path_expand()
ecg_paths <-
	fs::dir_ls(ecg_dir, glob = '*.ecgpuwave') |>
	fs::path_ext_remove()

# Annotation files from LUDB ----

rec <- '44'
rec_dir <- fs::path('~/OneDrive - University of Illinois Chicago/data/dsp/ludb/data/')
ann <- 'i'


ex <-
	read_wfdb(record = rec,
						record_dir = rec_dir,
						annotator = ann)

ggm(ex) |>
	draw_boundary_mask()
