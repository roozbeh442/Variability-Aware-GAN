clc
clear all; close all; clc;
%% Wasserstein Generative Adversarial Network
%% Load Data
%datasetFolder = './snesim_rels';
datasetFolder = 'C:\Matlab_C\img_rels_flat';
% datasetFolder = 'D:\SantosProject\direct_ti_rels';
%datasetFolder = 'E:\santosproject\img_rels_flat';
imds = imageDatastore(datasetFolder, ...
    'IncludeSubfolders',true);
augmenter = imageDataAugmenter('RandXReflection',false);
augimds = augmentedImageDatastore([100 100],imds,'DataAugmentation',augmenter);

load('prob_map_snesim_1000.mat');
dlref_prob_mat=dlarray(prob_map,'SSCB');

numLatentInputs = 100;
[dlnetD]=DiscriminatorW();
[dlnetG]=GeneratorW();

%% Setting
miniBatchSize = 32;
numIterationsG = 15000;
numIterationsDPerG = 5;
lambda = 10;
learnRateD = 2e-4; %2e-4
learnRateG = 1e-3;
gradientDecayFactor = 0;
squaredGradientDecayFactor = 0.9;
validationFrequency = 5;
%% Train
augimds.MiniBatchSize = miniBatchSize;
executionEnvironment = "auto";

mbq = minibatchqueue(augimds,...
    'MiniBatchSize',miniBatchSize,...
    'PartialMiniBatch','discard',...
    'MiniBatchFcn', @preprocessMiniBatch,...
    'MiniBatchFormat','SSCB',...
    'OutputEnvironment',executionEnvironment);

trailingAvgD = [];
trailingAvgSqD = [];
trailingAvgG = [];
trailingAvgSqG = [];

numValidationImages = 25;
ZValidation = randn([1 1 numLatentInputs numValidationImages],'single');
dlZValidation = dlarray(ZValidation,'SSCB');

if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
    dlZValidation = gpuArray(dlZValidation);
end

f = figure;
f.Position(3) = 2*f.Position(3);
imageAxes = subplot(1,2,1);
scoreAxes = subplot(1,2,2);
C = colororder;
lineLossD = animatedline(scoreAxes,'Color',C(1,:));
lineLossDUnregularized = animatedline(scoreAxes,'Color',C(2,:));
legend('With Gradient Penanlty','Unregularized')
xlabel("Generator Iteration")
ylabel("Discriminator Loss")
grid on

iterationG = 0;
iterationD = 0;
start = tic;

lossDV =-999;lossDUnregularizedV =-999;
    
% Loop over mini-batches
while iterationG < numIterationsG
    iterationG = iterationG + 1;

    % Train discriminator only
    for n = 1:numIterationsDPerG
        iterationD = iterationD + 1;

        % Reset and shuffle mini-batch queue when there is no more data.
        if ~hasdata(mbq)
            shuffle(mbq);
        end

        % Read mini-batch of data.
        dlX = next(mbq);

        % Generate latent inputs for the generator network. Convert to
        % dlarray and specify the dimension labels 'CB' (channel, batch).
        Z = randn([1 1 numLatentInputs size(dlX,4)],'like',dlX);
        dlZ = dlarray(Z,'SSCB');

        % Evaluate the discriminator model gradients using dlfeval and the
        % modelGradientsD function listed at the end of the example.
        [gradientsD, lossD, lossDUnregularized] = dlfeval(@modelGradientsD, dlnetD, dlnetG, dlX, dlZ, lambda);

        % Update the discriminator network parameters.
        [dlnetD,trailingAvgD,trailingAvgSqD] = adamupdate(dlnetD, gradientsD, ...
            trailingAvgD, trailingAvgSqD, iterationD, ...
            learnRateD, gradientDecayFactor, squaredGradientDecayFactor);
    end

    % Generate latent inputs for the generator network. Convert to dlarray
    % and specify the dimension labels 'CB' (channel, batch).
    Z = randn([1 1 numLatentInputs size(dlX,4)],'like',dlX);
    dlZ = dlarray(Z,'SSCB');
	
% 	if iterationG > 1500
%         lambda = 10;
%         learnRateD = 2e-4;
%         learnRateG = 1e-3;
%     end
		
    % Evaluate the generator model gradients using dlfeval and the
    % modelGradientsG function listed at the end of the example.
    gradientsG = dlfeval(@modelGradientsG, dlnetG, dlnetD, dlZ,dlref_prob_mat);

    % Update the generator network parameters.
    [dlnetG,trailingAvgG,trailingAvgSqG] = adamupdate(dlnetG, gradientsG, ...
        trailingAvgG, trailingAvgSqG, iterationG, ...
        learnRateG, gradientDecayFactor, squaredGradientDecayFactor);

    % Every validationFrequency generator iterations, display batch of
    % generated images using the held-out generator input
    if mod(iterationG,validationFrequency) == 0 || iterationG == 1
        % Generate images using the held-out generator input.
        dlXGeneratedValidation = predict(dlnetG,dlZValidation);

        % Tile and rescale the images in the range [0 1].
        I = imtile(extractdata(dlXGeneratedValidation));
        I = rescale(I);

        % Display the images.
        subplot(1,2,1);
        imagesc(I)
        xticklabels([]);
        yticklabels([]);
        title("Generated Images");
    end

    % Update the scores plot
    subplot(1,2,2)

    lossD = double(gather(extractdata(lossD)));
    lossDUnregularized = double(gather(extractdata(lossDUnregularized)));
    addpoints(lineLossD,iterationG,lossD);
    addpoints(lineLossDUnregularized,iterationG,lossDUnregularized);
    
    lossDV =[lossDV,lossD];
    lossDUnregularizedV =[lossDUnregularizedV,lossDUnregularized];
    
    D = duration(0,0,toc(start),'Format','hh:mm:ss');
    title( ...
        "Iteration: " + iterationG + ", " + ...
        "Elapsed: " + string(D))
    drawnow
    if mod(iterationG,50)==0
        save('timing.mat','lossDV','lossDUnregularizedV','iterationG','D');
        save(['VWGANgp-snesim',num2str(iterationG),'.mat']);
    end
end

save('VWGANgp-snesim.mat');

%% Helper Functions
%% preprocess
function X = preprocessMiniBatch(data)

% Concatenate mini-batch
X = cat(4,data{:});

% Rescale the images in the range [-1 1].
X = rescale(X,-1,1,'InputMin',0,'InputMax',255);

end

%% Generator
function [dlnetG]=GeneratorW()
filterSize = 5;
numFilters = 64;
numLatentInputs = 100;

projectionSize = [7 7 512];

layersG = [
    imageInputLayer([1 1 numLatentInputs],'Normalization','none','Name','in')
    projectAndReshapeLayer(projectionSize,numLatentInputs,'proj');
    transposedConv2dLayer(filterSize,4*numFilters,'Name','tconv1')
%     batchNormalizationLayer('Name','bnorm1')
    reluLayer('Name','relu1')
    transposedConv2dLayer(filterSize,2*numFilters,'Stride',2,'Name','tconv2')
%     batchNormalizationLayer('Name','bnorm2')
    reluLayer('Name','relu2')
    transposedConv2dLayer(filterSize,numFilters,'Stride',2,'Cropping','same','Name','tconv3')
    batchNormalizationLayer('Name','bnorm3')
    reluLayer('Name','relu3')
    transposedConv2dLayer(filterSize,1,'Stride',2,'Cropping','same','Name','tconv4')
    tanhLayer('Name','tanh4')];

    lgraphG = layerGraph(layersG);
    dlnetG = dlnetwork(lgraphG);
end
%% Discriminator
function [dlnetD]=DiscriminatorW()
numFilters = 64;
scale = 0.2;

inputSize = [100 100 1];
filterSize = 5;

layersD = [
    imageInputLayer(inputSize,'Normalization','none','Name','in')
    dropoutLayer(0.5,'Name','dropout')
    convolution2dLayer(filterSize,numFilters,'Stride',2,'Padding','same','Name','conv1')
    leakyReluLayer(scale,'Name','lrelu1')
    convolution2dLayer(filterSize,2*numFilters,'Stride',2,'Padding','same','Name','conv2')
%     batchNormalizationLayer('Name','bn2')
    leakyReluLayer(scale,'Name','lrelu2')
    convolution2dLayer(filterSize,4*numFilters,'Stride',2,'Padding','same','Name','conv3')
%     batchNormalizationLayer('Name','bn3')
    leakyReluLayer(scale,'Name','lrelu3')
    convolution2dLayer(filterSize,8*numFilters,'Stride',2,'Padding','same','Name','conv4')
    batchNormalizationLayer('Name','bn4')
    leakyReluLayer(scale,'Name','lrelu4')
    convolution2dLayer(7,1,'Name','conv5')];
%     sigmoidLayer('Name','sigmoid')

    lgraphD = layerGraph(layersD);
    dlnetD = dlnetwork(lgraphD);
end

%% modelGradients
function [gradientsD, lossD, lossDUnregularized] = modelGradientsD(dlnetD, dlnetG, dlX, dlZ, lambda)
% flipFactor = 0.5;
% Calculate the predictions for real data with the discriminator network.
dlYPred = forward(dlnetD, dlX);

% Calculate the predictions for generated data with the discriminator
% network.
dlXGenerated = forward(dlnetG,dlZ);
dlYPredGenerated = forward(dlnetD, dlXGenerated);

% Calculate the loss.
lossDUnregularized = mean(dlYPredGenerated - dlYPred);

% Calculate and add the gradient penalty. 
epsilon = rand([1 1 1 size(dlX,4)],'like',dlX);
dlXHat = epsilon.*dlX + (1-epsilon).*dlXGenerated;
dlYHat = forward(dlnetD, dlXHat);

% fliProb=randperm(size(dlYHat,4),floor(flipFactor*size(dlYHat,4)));
% dlYHat(:,:,:,fliProb) = 1-dlYHat(:,:,:,fliProb);
% Calculate gradients. To enable computing higher-order derivatives, set
% 'EnableHigherDerivatives' to true.
gradientsHat = dlgradient(sum(dlYHat),dlXHat,'EnableHigherDerivatives',true);
gradientsHatNorm = sqrt(sum(gradientsHat.^2,1:3) + 1e-10);
gradientPenalty = lambda.*mean((gradientsHatNorm - 1).^2);

% Penalize loss.
lossD = lossDUnregularized + gradientPenalty;

% Calculate the gradients of the penalized loss with respect to the
% learnable parameters.
gradientsD = dlgradient(lossD, dlnetD.Learnables);

end

function gradientsG =  modelGradientsG(dlnetG, dlnetD, dlZ,dlref_prob_mat)

% Calculate the predictions for generated data with the discriminator
% network.
dlXGenerated = forward(dlnetG,dlZ);
dlYPredGenerated = forward(dlnetD, dlXGenerated);

% calculate the norm distance between pob map pf the batch and ref
dlXGenerated=rescale(dlXGenerated,0,1,'InputMax',1,'InputMin',-1);
batch_prob_mat=sum(dlXGenerated,4)./size(dlXGenerated,4);

% probability loss
Loss_prob=sum((abs(batch_prob_mat-dlref_prob_mat)),'all')/100;
alpha=1.0;

% Calculate the loss.
lossG = -mean(dlYPredGenerated)+(alpha)*Loss_prob;

% Calculate the gradients of the loss with respect to the learnable
% parameters.
gradientsG = dlgradient(lossG, dlnetG.Learnables);

end
