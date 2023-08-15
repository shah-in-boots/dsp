# Libraries
library(fs)
library(shiva)

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
dat1 <- read_wfdb(record = '1', record_dir = ludb)
hea1 <- read_header(record = '1', record_dir = ludb)
ann1 <- read_annotation(record = '1', record_dir = ludb, annotator = 'i')


