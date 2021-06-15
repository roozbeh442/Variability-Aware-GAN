function [lossGenerator, lossDiscriminator,Loss_prob] = ganLoss(probReal,probGenerated,batch_prob_mat,dlref_prob_mat)

% Calculate the loss for the discriminator network.
lossDiscriminator =  -mean(log(probReal)) -mean(log(1-probGenerated));

% probability loss
Loss_prob=sum((abs(batch_prob_mat-dlref_prob_mat)),'all')/10000;
alpha=1.0;
% Calculate the loss for the generator network.
lossGenerator = -mean(log(probGenerated))+(alpha)*Loss_prob;

end