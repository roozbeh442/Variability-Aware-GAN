
dataSetFolder = fullfile(location);
imds=imageDatastore(dataSetFolder,'IncludeSubfolders',true, 'LabelSource','foldernames');

classes = categories(imds.Labels);
numClasses=numel(classes);
pause(0.5)

augmenter = imageDataAugmenter ('RandXReflection',false);
augimds = augmentedImageDatastore([100 100],imds,'DataAugmentation',augmenter);
% augmenter = imageDataAugmenter ('RandXReflection',false);
% augimds = augmentedImageDatastore([100 100],imds,'DataAugmentation',augmenter);

Generator;

Discriminator;

