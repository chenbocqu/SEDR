

%% 降维，稀疏表示联合优化
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

%% 设置实验参数
rdim            = 169;  % 降维维度 PCA，SRDR
isdr            = true; % 是否降维
issr            = true; % 是否用字典学习

%% SVM学习率
lambda          = 0.03;

%% SR参数
para.p              = rdim;      % 降维的维数
para.K              = 100;

para.lambda1        = 3e-2;     % lambda1 ||S||_1
para.lambda2        = 1.2;
para.MaxIters       = 25;

para.draw           = true;

SC_param.mode       = 2;
SC_param.lambda     = para.lambda1;
% SC_param.lambda2    = para.lambda1;

para.sc_para        = SC_param;

reco_rates= [];
% 61.23
for dataset=[4]
    
    %% 加载数据
    switch dataset
        case 1
            load YaleB_DR_DAT
            f = 10; % f
            dname = 'Extended Yale B dataset';
            para.K              = 380;
            
        case 2
            load AR_DR_DAT
            f = 10;
            dname = 'AR dataset';
            para.K              = 500;
            
        case 3
            load MNIST
            f = 10;
            dname = 'MNIST dataset';
            para.K              = 300;
        case 4
            load USPS
            f = 60;
            dname = 'USPS dataset';
            para.K              = 300;
            
        otherwise
            error('Unknown dataset.')
    end
    
    tr_dat  = Train_DAT;
    tt_dat  = Test_DAT;
    trls    = trainlabels;
    ttls    = testlabels;
    
    clear Train_DAT Test_DAT trainlabels testlabels;
    
    %% 数据归一化
    X_train         = normalize_mat(tr_dat);
    X_test          = normalize_mat(tt_dat);
    
    
    [ P,B,X_train,J ]     = pca_sr_dr( X_train,para );
    
    X_test          = mexLasso( P*X_test,B,SC_param ); 
    
    %% SVM训练   
    [w, b, class_name]  = li2nsvm_multiclass_lbfgs(X_train', trls', lambda);
    
    [ttls_pred, ~]      = li2nsvm_multiclass_fwd(X_test', w, b, class_name);
    reco_rate           = (sum(ttls_pred'==ttls))/length(ttls);
    
    disp(['Recognition rate on the ', dname, ' is ', num2str(roundn(reco_rate*100,-3)) '%']);
    reco_rates = [reco_rates; roundn(reco_rate*100,-3) ];
    
    %% 绘制低维字典和重构字典
    mdl.P=P;mdl.B=B;
    plot_dict(mdl);
end

reco_rates

% OOPDR + SVM：93.622%
% PCA+SR+SVM：94.569%（ r=196, K=350 ）



