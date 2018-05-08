

%% 降维，稀疏表示步骤独立时

clear all; clc;
warning off;

%% load data
addpath('.\data');

%% load tool
addpath('.\large_scale_svm');
addpath('.\dictionary_learning');
addpath('.\FOptM');
addpath('.\common_tool');
addpath('.\mylib');
addpath('.\test');

% dataset = {'YaleB_DR_DAT','AR_DR_DAT'};
% n       = size(dataset,2);

%% 设置实验参数
rdim            = 169;  % 降维维度 PCA，SRDR
isdr            = true; % 是否降维
issr            = false; % 是否用字典学习

%% SVM学习率
lambda          = 0.03;

%% SR参数
param.K         = 100; % learns a dictionary with 100 elements
param.lambda    = 0.03;
param.numThreads= -1; % number of threads
param.batchsize = 400;
param.verbose   = false;

param.iter      = 100; % let us see what happens after 1000 iterations.

reco_rates= [];

% param.lambda = 0.03 : 1-2 ; 0.15 : 3-4
for dataset=[1]
    
    %% 加载数据
    switch dataset
        case 1
            load YaleB_DR_DAT
            f = 10; % f
            dname = 'Extended Yale B dataset';
            param.K         = 380;
        case 2
            load AR_DR_DAT
            f = 10;
            dname = 'AR dataset';
            param.K         = 500;
            
        case 3
            load MNIST
            f = 10;
            dname = 'MNIST dataset';
            param.K         = 300;
        case 4
            load USPS
            f = 60;
            dname = 'USPS dataset';
            param.K         = 300;
            
        case 5
            load Fault_DATA
            f = 60;
            dname = 'Fault Diagnosis';
            param.K         = 300;
            
        otherwise
            error('Unknown dataset.')
    end
    
    tr_dat = Train_DAT;
    tt_dat = Test_DAT;
    trls = trainlabels;
    ttls = testlabels;
    
    clear Train_DAT Test_DAT trainlabels testlabels;
    
    %% 数据归一化
    X_train         = normalize_mat(tr_dat);
    X_test          = normalize_mat(tt_dat);
    
    %% 1. PCA + SVM，先降维 
    if isdr
    M               = Eigenface_f(X_train,rdim); % P = M'
    
    X_train         = M'*X_train;
    X_test          = M'*X_test;
    end
    
    if issr
        D               = mexTrainDL(X_train,param);
        X_train         = mexLasso(X_train,D,param);
        X_test          = mexLasso(X_test,D,param);
        
%         %% 可视化字典
%         figure;ImD=displayPatches(D);      % 可视化字典
%         imagesc(ImD); colormap('gray');
    end
    
    %% SVM训练
    tic;
    [w, b, class_name]  = li2nsvm_multiclass_lbfgs(X_train', trls', lambda);
    toc;
    tic;
    [ttls_pred, ~]      = li2nsvm_multiclass_fwd(X_test', w, b, class_name);
    reco_rate           = (sum(ttls_pred'==ttls))/length(ttls);
    toc;
    disp(['Recognition rate on the ', dname, ' is ', num2str(roundn(reco_rate*100,-3)) '%']);
    reco_rates = [reco_rates; roundn(reco_rate*100,-3) ];
    
%     my_draw_cm;
%     mdl.P=M';mdl.B=D;
%     plot_dict(mdl);
end

reco_rates



