library(tidyverse)
library(xml2)

doc <- read_xml("lspro/sample_data/PatientData V2_7.xsd")
strx <- xml_structure(doc)

doc <- read_xml("lspro/sample_data/PatientData Export.xml")
strx <- xml_structure(doc)
