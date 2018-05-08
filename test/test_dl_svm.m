
load USPS
dname = 'USPS';

X           = normalize_mat(Train_DAT);
X_test      = normalize_mat(Test_DAT);

param.K         = 100; % learns a dictionary with 100 elements
param.lambda    = 0.15;
param.numThreads= -1; % number of threads
param.batchsize = 400;
param.verbose   = false;

param.iter      = 25; % let us see what happens after 1000 iterations.

tic
D               = mexTrainDL(X,param);
t=toc;

S               = mexLasso(X,D,param);

% %% SVM 训练
% mdl             = train(trainlabels', S' );
% 
% %% 测试
S_test          = mexLasso(X_test,D,param);
% [predicted_label, accuracy, dp] = predict(testlabels', S_test', mdl);

lambda              = 0.3;
[w, b, class_name]  = li2nsvm_multiclass_lbfgs(S', trainlabels', lambda);

[ttls_pred, ~]      = li2nsvm_multiclass_fwd(S_test', w, b, class_name);
reco_rate           =  (sum(ttls_pred'==testlabels))/length(testlabels);

disp(['Recognition rate on the ', dname, ' is ', num2str(roundn(reco_rate*100,-3)) '%']);

%% 可视化字典
figure;
ImD=displayPatches(D);      % 可视化字典
imagesc(ImD); colormap('gray');

