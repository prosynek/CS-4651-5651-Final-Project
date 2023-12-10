load('resnet50_translearn_spec_3.mat')   % load trained network

%%
target_size = [224,224];                 % [227, 227] for squeezenet
resize_fnc = @(x) imresize(x, target_size);
testDatastore = transform(testDatastore, resize_fnc);

%%
[y_pred, scores] = classify(trainedNetwork_1, testDatastore);

%%
testLabels = testDatastore.UnderlyingDatastores{1,1}.Labels;

%%
cm = confusionchart(testLabels, y_pred)

%%
roc_obj = rocmetrics(testLabels, scores, {'bonafide', 'spoof'})
plot(roc_obj)