% Attempt to read XMLS fileo

% Pull in generic structure
sample_data = [".." filesep ".." filesep ".." filesep "One

	/Users/asshah4/OneDrive - University of Illinois at Chicago/data/signals

	/Users/asshah4/projects/signals/lspro


S = readstruct("sample_data/PatientData Export.xml");

% Patient information
patient_info = S.Patient.PatientInfo;

% Recording information stored here
R = S.Patient.RecordingList;
rec_file = R.Recording.FileName;
rec_id = R.Recording.idAttribute;
rec_start = R.Recording.StartTime;
rec_stop = R.Recording.StopTime;

% Data and class information
% There may be multiple logs or files here
% Will pick the most "recent" file
D = S.Patient.Children.Child;

max(D.timeAttribute)
