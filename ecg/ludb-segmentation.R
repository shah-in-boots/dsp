# Libraries
library(fs)
library(shiva)
library(ggplot2)

# Data location
data_loc <- function() {
	x <- sessionInfo()$running
	
	if (grepl('linux', x)) {
		file.path('/Users/asshah4/OneDrive - University of Illinois Chicago/data')
	} else if (grepl('mac', x)) {
		file.path('/Users/asshah4/OneDrive - University of Illinois Chicago/data')
	}
	
}

# Multiple lead data from LUDB
# 	200 ECGs named sequentially from 1:200
# 	Each one has a *.dat file, a *.hea file, and an annotation for all leads
ludb <- fs::path(data_loc(), 'dsp', 'ludb', 'data')

# Example from patient 1
dat_1 <- read_wfdb(record = '1', record_dir = ludb)
hea_1 <- read_header(record = '1', record_dir = ludb)
ann_i_1 <- read_annotation(record = '1', record_dir = ludb, annotator = 'i')

ecg <- egm(dat_1, hea_1, ann_i_1)

ggm(ecg, channels = hea_1$label) |>
	draw_boundary_mask()

# Labeled data
# 	Take each P, QRS, T wave region and mask that onto original signal
# 	This would be similar to "one hot encoding" for each
# 	Would also needed a "background" or ZERO label

# Input will be 5000 x k (10 sec samples @ 500 hertz; k = number of leads)
# 	Testing data will be annotated 5000 x 12 x 4 (4 = the encodings)
# Output will be similar as a seq2seq problem
# 	5000 x k x 4 (4 = encoding/predictions or labels)
