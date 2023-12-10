% 70/10/20 Train/Val/Test Split
asvDatastore = imageDatastore('D:\CSC 5651\asvspoof_mel_spectrograms_new', IncludeSubfolders=true, FileExtensions='.png', LabelSource='foldernames')
[trainDatastore, testDatastore] = splitEachLabel(asvDatastore, 0.8)
% Use 13 percent of train datastore for validation when importing data in deep network designer