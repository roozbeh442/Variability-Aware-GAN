% location='./direct_ti_rels';
location='C:\Matlab_C\direct_ti_rels';
load('prob_map_direct_1000.mat');
%ref_prob_mat=ones(size(ref_prob_mat))*0.30;
dlref_prob_mat=dlarray(prob_map,'SSCB');
data_organizer
%=============================================================
numEpochs = 1000;
miniBatchSize = 142;
augimds.MiniBatchSize = miniBatchSize;

learnRate = 0.0002;
gradientDecayFactor = 0.5;
squaredGradientDecayFactor = 0.999;

flipFactor = 0.3;
validationFrequency = 100;

trailingAvgGenerator = [];
trailingAvgSqGenerator = [];
trailingAvgDiscriminator = [];
trailingAvgSqDiscriminator = [];

numValidationImages = 25;
ZValidation = randn(1,1,numLatentInputs,numValidationImages,'single');
dlZValidation = dlarray(ZValidation,'SSCB');

executionEnvironment = "cpu";
iteration = 0;
start = tic;

f = figure;
f.Position(3) = 2*f.Position(3);
imageAxes = subplot(1,2,1);
scoreAxes = subplot(1,2,2);
lineScoreGenerator = animatedline(scoreAxes,'Color',[0 0.447 0.741]);
lineScoreDiscriminator = animatedline(scoreAxes, 'Color', [0.85 0.325 0.098]);
legend('Generator','Discriminator');
ylim([0 1])
xlabel("Iteration")
ylabel("Score")
grid on
lossGeneratorV=-999; lossDiscriminatorV=-999; Loss_probV=-999;
% Loop over epochs.
for epoch = 1:numEpochs
    
    % Reset and shuffle datastore.
    reset(augimds);
    augimds = shuffle(augimds);
    
    % Loop over mini-batches.
    while hasdata(augimds)
        iteration = iteration + 1;
        
        % Read mini-batch of data.
        data = read(augimds);
        
        % Ignore last partial mini-batch of epoch.
        if size(data,1) < miniBatchSize
            continue
        end
        
        % Concatenate mini-batch of data and generate latent inputs for the
        % generator network.
        X = cat(4,data{:,1}{:});
        X = single(X);
        Z = randn(1,1,numLatentInputs,size(X,4),'single');
        
        % Rescale the images in the range [-1 1].
        X = rescale(X,-1,1,'InputMin',0,'InputMax',255);
        
        % Convert mini-batch of data to dlarray and specify the dimension labels
        % 'SSCB' (spatial, spatial, channel, batch).
        dlX = dlarray(X, 'SSCB');
        dlZ = dlarray(Z, 'SSCB');
        
        % If training on a GPU, then convert data to gpuArray.
        if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
            dlX = gpuArray(dlX);
            dlZ = gpuArray(dlZ);
        end
        
        % Evaluate the model gradients and the generator state using
        % dlfeval and the modelGradients function listed at the end of the
        % example.
        [gradientsGenerator, gradientsDiscriminator, stateGenerator, scoreGenerator, scoreDiscriminator,lossGenerator, lossDiscriminator,Loss_prob] = ...
            dlfeval(@modelGradients, dlnetGenerator, dlnetDiscriminator, dlX, dlZ, flipFactor,dlref_prob_mat);
        dlnetGenerator.State = stateGenerator;

        lossGeneratorV=[lossGeneratorV,lossGenerator];
        lossDiscriminatorV=[lossDiscriminatorV,lossDiscriminator];
        Loss_probV=[Loss_probV,Loss_prob];

        % Update the discriminator network parameters.
        [dlnetDiscriminator,trailingAvgDiscriminator,trailingAvgSqDiscriminator] = ...
            adamupdate(dlnetDiscriminator, gradientsDiscriminator, ...
            trailingAvgDiscriminator, trailingAvgSqDiscriminator, iteration, ...
            learnRate, gradientDecayFactor, squaredGradientDecayFactor);
        
        % Update the generator network parameters.
        [dlnetGenerator,trailingAvgGenerator,trailingAvgSqGenerator] = ...
            adamupdate(dlnetGenerator, gradientsGenerator, ...
            trailingAvgGenerator, trailingAvgSqGenerator, iteration, ...
            learnRate, gradientDecayFactor, squaredGradientDecayFactor);
%         save('timing.mat','iteration','lossGenerator','lossDiscriminator','epoch');
        % Every validationFrequency iterations, display batch of generated images using the
        % held-out generator input
        if mod(iteration,validationFrequency) == 0 || iteration == 1
            % Generate images using the held-out generator input.
            dlXGeneratedValidation = predict(dlnetGenerator,dlZValidation);
            
            % Tile and rescale the images in the range [0 1].
            I = imtile(extractdata(dlXGeneratedValidation));
            I = rescale(I);
            
            % Display the images.
            subplot(1,2,1);
            image(imageAxes,I)
            xticklabels([]);
            yticklabels([]);
            title("Generated Images");
        end
        
        % Update the scores plot
        subplot(1,2,2)
        addpoints(lineScoreGenerator,iteration,...
            double(gather(extractdata(scoreGenerator))));
        
        addpoints(lineScoreDiscriminator,iteration,...
            double(gather(extractdata(scoreDiscriminator))));
        
        % Update the title with training progress information.
        D = duration(0,0,toc(start),'Format','hh:mm:ss');
        title(...
            "Epoch: " + epoch + ", " + ...
            "Iteration: " + iteration + ", " + ...
            "Elapsed: " + string(D))
        
        drawnow
    end
    save('timing.mat','iteration','lossGeneratorV','lossDiscriminatorV','Loss_probV','epoch');
	if mod(epoch,50)==0
		save('VDCGAN_snesim_rels_Z150.mat')
	end
	
end

save('VDCGAN_snesim_rels_Z150.mat')