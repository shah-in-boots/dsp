library(shiva)
library(fs)

home <- fs::path_expand("~")
main <- fs::path("projects", "dsp")
folder <- fs::path(home, main, "ecg")

filePaths <- fs::dir_ls(path = folder, glob = "*.xml")
fileNames <- fs::path_file(filePaths) |> fs::path_ext_remove()

for (i in seq_along(filePaths)) {
	
	ecg <- shiva::read_muse(filePaths[i])
	sig <- vec_data(ecg)
	hea <- attr(ecg, "header")
	
	shiva::write_wfdb(
		data = sig,
		type = "muse",
		record = fileNames[i],
		record_dir = folder,
		wfdb_path = "/shared/home/ashah282/bin",
		header = hea
	)

}
