%set seed
rng(1234);
sample_train = datasample(1:n,483186,'Replace',false);

sample_X=(X(sample_train,:));
sample_Y=(target(sample_train,:));

% complete_data=[target X];
% csvwrite('newdat',complete_data);

sample_test=setdiff(1:n,sample_train);
summary(categorical(sample_Y))
%      1       30418
%      2       16846 
%      3      336903  
%      4       40628 
%      5       58391 

I1=find(sample_Y==1);
I2=find(sample_Y==2);
I3=find(sample_Y==3);
I4=find(sample_Y==4);
I5=find(sample_Y==5);

%Take representative Samples
sample_train1 = datasample(I1,15000,'Replace',false);
sample_train2 = datasample(I2,15000,'Replace',false);
sample_train3 = datasample(I3,15000,'Replace',false);
sample_train4 = datasample(I4,15000,'Replace',false);
sample_train5 = datasample(I5,15000,'Replace',false);
sample_trainnew=[sample_train1;sample_train2;sample_train3;sample_train4;sample_train5];

%Take thh reminder
sample_test1=setdiff(1:size(sample_train,2),sample_train1);
sample_test2=setdiff(sample_test1,sample_train2);
sample_test3=setdiff(sample_test2,sample_train3);
sample_test4=setdiff(sample_test3,sample_train4);
sample_testnew=setdiff(sample_test4,sample_train5);

Xtrain_new=X(sample_trainnew,:);
Ytrain_new=target(sample_trainnew);

Xtest_new=X(sample_testnew,:);
Ytest_new=target(sample_testnew);
%Take unbalanced sample
%Take representative Samples
sample_trainub1 = datasample(I1,4727,'Replace',false);
sample_trainub2 = datasample(I2,2608,'Replace',false);
sample_trainub3 = datasample(I3,52273,'Replace',false);
sample_trainub4 = datasample(I4,6304,'Replace',false);
sample_trainub5 = datasample(I5,9086,'Replace',false);
sample_trainubnew=[sample_trainub1;sample_trainub2;sample_trainub3;sample_trainub4;sample_trainub5];

%Take thh reminder
sample_testub1=setdiff(1:size(sample_train,2),sample_trainub1);
sample_testub2=setdiff(sample_testub1,sample_trainub2);
sample_testub3=setdiff(sample_testub2,sample_trainub3);
sample_testub4=setdiff(sample_testub3,sample_trainub4);
sample_testubnew=setdiff(sample_testub4,sample_trainub5);

Xtrain_ubnew=X(sample_trainubnew,:);
Ytrain_ubnew=target(sample_trainubnew);

Xtest_ubnew=X(sample_testubnew,:);
Ytest_ubnew=target(sample_testubnew);

cvfit_glmnet = cvglmnet(Xtrain_new, Ytrain_new,'multinomial');
cvglmnetPlot(cvfit_glmnet);

cvfit_glmnetfinal = cvglmnet(Xtrain, Ytrain,'multinomial');
opt.lambda_min=[0.0002];
fit_glmnetfinal=glmnet(X,target,'multinomial');

%cvfit_glmnet.glmnet_fit.beta{2}(:,67)
reduced_pred=find(cvfit_glmnet.glmnet_fit.beta{2}(:,67));
nnz(reduced_pred); 
%61
% reduced_X=X(:,reduced_pred);
% Xtrain=(reduced_X(sample_train,:));
% % Ytrain=(target(sample_train,:));
% sample_test=setdiff(1:n,sample_train);
% Xtest=(reduced_X(sample_test,:));
% Ytest=(target(sample_test,:));
%equal representative sample
Mdl = fitcnb(Xtrain_new,Ytrain_new);
label = predict(Mdl,Xtest_new);
%labelp=posterior(Mdl,Xtest);
sum(Ytest_new==label)
% 109363
labeln=label-1;
labelp=label+1;
sum(Ytest_new==labeln)
%      51071

sum(Ytest_new==labelp)
%   52105

Mdlub = fitcnb(Xtrain_ubnew,Ytrain_ubnew);
labelub = predict(Mdlub,Xtest_ubnew);
%labelp=posterior(Mdl,Xtest);
sum(Ytest_ubnew==labelub)
%   106732

rng(1234); % For reproducibility
CVMdl = crossval(Mdl,'KFold',5);
CVMdl_k=fitcnb(Xtrain_new,Ytrain_new,'KFold',5);
labelCV = predict(CVMdl_k,Xtest);
modelCVLoss = kfoldLoss(CVMdl_k);
testingLoss1 = loss(CVMdl_k.Trained{1},Xtest_new,Ytest_new);
testingLoss2 = loss(CVMdl_k.Trained{2},Xtest_new,Ytest_new);
testingLoss3 = loss(CVMdl_k.Trained{3},Xtest_new,Ytest_new);
testingLoss4 = loss(CVMdl_k.Trained{4},Xtest_new,Ytest_new);
testingLoss5 = loss(CVMdl_k.Trained{5},Xtest_new,Ytest_new);
testInds = test(CVMdl_k.Partition);

 
%loss_new = loss(Mdl_new.Trained{1},Xtest_new,Ytest_new);
label_new=predict(Mdl_new,Xtest_new);
testingLossnew1 = loss(Mdl_new.Trained{1},Xtest_new,Ytest_new);
testingLossnew2 = loss(Mdl_new.Trained{2},Xtest_new,Ytest_new);
testingLossnew3 = loss(Mdl_new.Trained{3},Xtest_new,Ytest_new);
testingLossnew4 = loss(Mdl_new.Trained{4},Xtest_new,Ytest_new);
testingLossnew5 = loss(Mdl_new.Trained{5},Xtest_new,Ytest_new);

% testingLossn = loss(CVMdl_k.Trained{1},Xtest,Ytestn);
% testingLossp = loss(CVMdl_k.Trained{1},Xtest,Ytestp);
% labelCV=predict(CVMdl.Trained,Xtest);
% [labelCV,score] = kfoldPredict(CVMdl);
[B_mnr,dev,stats] = mnrfit(Xtrain_new,Ytrain_new,'model','nominal');
%label_mnr=predict(B_mnr,Ytest);
[pihat,dlow,dhi] = mnrval(B_mnr,Xtest_new,stats,'model','nominal');
[M,I] = max(pihat,[],2);
summary(categorical(I))
summary(categorical(Ytest_new))
sum((I)==Ytest_new)
traindata=[Xtrain Ytrain];


 b = TreeBagger(10,Xtrain_new,Ytrain_new,'Method','classification','OOBPred','On');
label_new=str2num(predict(b,Xtest_new));
S = sprintf('%s ', label_new{:});
D = sscanf(S, '%f');

sum(D==Ytest_new)
