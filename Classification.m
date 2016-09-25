%% Take the training, validation and the test sets
rng(1234);

% X constains predictors and Y contains target
%training and validation together
sample_train = datasample(1:n,483186,'Replace',false);
sample_test = setdiff(1:n,sample_train);

% Initial Exploration of the clusters
%summary(categorical(Y))
%      1       30418
%      2       16846
%      3      336903
%      4       40628
%      5       58391

%Taking clusterwise indices
% I1=find(Y==1);
% I2=find(Y==2);
% I3=find(Y==3);
% I4=find(Y==4);
% I5=find(Y==5);

%% Take equal Representative Sample
sample_train1 = datasample(I1,15000,'Replace',false);
sample_train2 = datasample(I2,15000,'Replace',false);
sample_train3 = datasample(I3,15000,'Replace',false);
sample_train4 = datasample(I4,15000,'Replace',false);
sample_train5 = datasample(I5,15000,'Replace',false);
sample_trainnew=[sample_train1;sample_train2;sample_train3;sample_train4;sample_train5];

%Take the reminder
sample_test1=setdiff(1:size(sample_train,2),sample_train1);
sample_test2=setdiff(sample_test1,sample_train2);
sample_test3=setdiff(sample_test2,sample_train3);
sample_test4=setdiff(sample_test3,sample_train4);
sample_testnew=setdiff(sample_test4,sample_train5);

Xtrain_new=X(sample_trainnew,:);
Ytrain_new=target(sample_trainnew);

Xtest_new=X(sample_testnew,:);
Ytest_new=target(sample_testnew);

%% Actual Representative Sample
sample_trainub1 = datasample(I1,4727,'Replace',false);
sample_trainub2 = datasample(I2,2608,'Replace',false);
sample_trainub3 = datasample(I3,52273,'Replace',false);
sample_trainub4 = datasample(I4,6304,'Replace',false);
sample_trainub5 = datasample(I5,9086,'Replace',false);
sample_trainubnew=[sample_trainub1;sample_trainub2;sample_trainub3;sample_trainub4;sample_trainub5];

%Take the reminder
sample_testub1=setdiff(1:size(sample_train,2),sample_trainub1);
sample_testub2=setdiff(sample_testub1,sample_trainub2);
sample_testub3=setdiff(sample_testub2,sample_trainub3);
sample_testub4=setdiff(sample_testub3,sample_trainub4);
sample_testubnew=setdiff(sample_testub4,sample_trainub5);

Xtrain_ubnew=X(sample_trainubnew,:);
Ytrain_ubnew=target(sample_trainubnew);

Xtest_ubnew=X(sample_testubnew,:);
Ytest_ubnew=target(sample_testubnew);

%% Data reduction
cvfit_glmnet = cvglmnet(Xtrain_new, Ytrain_new,'multinomial');
cvglmnetPlot(cvfit_glmnet);

cvfit_glmnet = cvglmnet(Xtrain_ubnew, Ytrain_ubnew,'multinomial');
cvglmnetPlot(cvfit_glmnet);

cvfit_glmnetfinal = cvglmnet(Xtrain, Ytrain,'multinomial');
opt.lambda_min=[0.0002];
fit_glmnetfinal=glmnet(X,target,'multinomial', opt);

reduced_pred=find(cvfit_glmnet.glmnet_fit.beta{2}(:,67));
nnz(reduced_pred);

%% Naive Bayes

% Equal representative sample
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

%Actual Representative Sample
Mdlub = fitcnb(Xtrain_ubnew,Ytrain_ubnew);
labelub = predict(Mdlub,Xtest_ubnew);
%labelp=posterior(Mdl,Xtest);
sum(Ytest_ubnew==labelub)
%   106732

%Cross Validation equal representative sample
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

%Cross Validation actual representative sample
rng(1234); % For reproducibility
CVMdlub = crossval(Mdlub,'KFold',5);
CVMdl_kub=fitcnb(Xtrain_new,Ytrain_new,'KFold',5);
labelCVub = predict(CVMdl_kub,Xtest);
modelCVLoss = kfoldLoss(CVMdl_kub);
testingLossub1 = loss(CVMdl_kub.Trained{1},Xtest_new,Ytest_new);
testingLossub2 = loss(CVMdl_kub.Trained{2},Xtest_new,Ytest_new);
testingLossub3 = loss(CVMdl_kub.Trained{3},Xtest_new,Ytest_new);
testingLossub4 = loss(CVMdl_kub.Trained{4},Xtest_new,Ytest_new);
testingLossub5 = loss(CVMdl_kub.Trained{5},Xtest_new,Ytest_new);

%% Random Forest
%% bootstrap size 66% equal
b = TreeBagger(100,Xtrain_new,Ytrain_new,'Method','classification','OOBPred','On','Fboot',0.66);
label_new=(predict(b,Xtest_new));
S = sprintf('%s ', label_new{:});
D = sscanf(S, '%f');
sum(D==Ytest_new)/size(Ytest_new,1)
% 0.7433
%% bootstrap size 66% actual
b_ub = TreeBagger(100,Xtrain_ubnew,Ytrain_ubnew,'Method','classification','OOBPred','On','Fboot',0.66);
label_ubnew=(predict(b_ub,Xtest_ubnew));
Sub = sprintf('%s ', label_ubnew{:});
Dub = sscanf(Sub, '%f');
sum(Dub==Ytest_ubnew)/size(Ytest_ubnew,1)
% 0.7428
%% bootstrap size 75% equal
b = TreeBagger(100,Xtrain_new,Ytrain_new,'Method','classification','OOBPred','On','Fboot',0.75);
label_new=(predict(b,Xtest_new));
S = sprintf('%s ', label_new{:});
D = sscanf(S, '%f');
sum(D==Ytest_new)/size(Ytest_new,1)
%0.7459
%% bootstrap size 75% Actual
b_ub = TreeBagger(100,Xtrain_ubnew,Ytrain_ubnew,'Method','classification','OOBPred','On','Fboot',0.75);
label_ubnew=(predict(b_ub,Xtest_ubnew));
Sub = sprintf('%s ', label_ubnew{:});
Dub = sscanf(Sub, '%f');
sum(Dub==Ytest_ubnew)/size(Ytest_ubnew,1)
% 0.7455
%% bootstrap size 85% equal
b = TreeBagger(100,Xtrain_new,Ytrain_new,'Method','classification','OOBPred','On','Fboot',0.85);
label_new=(predict(b,Xtest_new));
S = sprintf('%s ', label_new{:});
D = sscanf(S, '%f');
sum(D==Ytest_new)/size(Ytest_new,1)
% 0.7472
%% bootstrap size 85% actual
b_ub = TreeBagger(100,Xtrain_ubnew,Ytrain_ubnew,'Method','classification','OOBPred','On','Fboot',0.85);
label_ubnew=(predict(b_ub,Xtest_ubnew));
Sub = sprintf('%s ', label_ubnew{:});
Dub = sscanf(Sub, '%f');
sum(Dub==Ytest_ubnew)/size(Ytest_ubnew,1)
% 0.7465

%% To choose Out-of-Bag Classification Error equal
plot(oobError(b))
xlabel('Number of Grown Trees')
ylabel('Out-of-Bag Classification Error')
title('OOB vs Trees - Error Rate, Equal Representative')
%% To choose Out-of-Bag Classification Error actual
plot(oobError(b_ub))
xlabel('Number of Grown Trees')
ylabel('Out-of-Bag Classification Error')
title('OOB vs Trees - Error Rate, Actual Representative')
%% Training a classification tree with the original training set after the parameters are chosen through CV - for tree bagger no explicit CV methods were used

rng(1234);
b_cv = TreeBagger(40,Xtrain,Ytrain,'Method','classification','OOBPred','On','OOBVarImp','On','Fboot',0.85);
disp([b_cv.OOBPermutedVarDeltaError]);
%% Prediction
[label_new,scores]=(predict(b_cv,Xtest));
S = sprintf('%s ', label_new{:});
D = sscanf(S, '%f');
sum(D==Ytest)/size(Ytest,1)
% 81.37% of the observations in the test set are classified correctly

%% Margin OOB
mar = meanMargin(b_cv,Xtest,Ytest)
plot(mar)
xlabel('Number of Grown Trees')
ylabel('Cumulative Out-of-Bag Mean Classification Margin')
title('Cumulative OOB Margin')

%% Variable Improtance
figure
bar(b_cv.OOBPermutedVarDeltaError)
xlabel 'Feature Number'
ylabel 'Out-of-Bag Feature Importance'
title 'Feature Importance'
xlim([0 102])
sum(b_cv.OOBPermutedVarDeltaError)
%478.3
idxvar = find(b_cv.OOBPermutedVarDeltaError>=7)
idxCategorical = find(iscategorical(idxvar)==1);
%% Error
err_cv = error(b_cv,Xtest,Ytest);
1-mean(err_cv)
%    0.8111

