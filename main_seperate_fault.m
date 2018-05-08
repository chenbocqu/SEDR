

%% ��ά��ϡ���ʾ�������ʱ

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

%% ����ʵ�����
rdim            = 300;  % ��άά�� PCA��SRDR
isdr            = true; % �Ƿ�ά
issr            = false; % �Ƿ����ֵ�ѧϰ

%% SVMѧϰ��
lambda          = 0.03;

%% SR����
param.K         = 100; % learns a dictionary with 100 elements
param.lambda    = 0.03;
param.numThreads= -1; % number of threads
param.batchsize = 400;
param.verbose   = false;

param.iter      = 100; % let us see what happens after 1000 iterations.

reco_rates= [];

% param.lambda = 0.03 : 1-2 ; 0.15 : 3-4
for dataset=[1]
    
    %% ��������
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
            param.K         = 500;
            
        otherwise
            error('Unknown dataset.')
    end
    
    tr_dat = Train_DAT;
    tt_dat = Test_DAT;
    trls = trainlabels;
    ttls = testlabels;
    
    clear Train_DAT Test_DAT trainlabels testlabels;
    
    tic;
    
    %% ���ݹ�һ��
    X_train         = normalize_mat(tr_dat);
    X_test          = normalize_mat(tt_dat);
    
    %% 1. PCA + SVM���Ƚ�ά 
    if isdr
    M               = Eigenface_f(X_train,rdim); % P = M'
    
    X_train         = M'*X_train;
    X_test          = M'*X_test;
    end
    
    if issr
        D               = mexTrainDL(X_train,param);
        X_train         = mexLasso(X_train,D,param);
        X_test          = mexLasso(X_test,D,param);
        
%         %% ���ӻ��ֵ�
%         figure;ImD=displayPatches(D);      % ���ӻ��ֵ�
%         imagesc(ImD); colormap('gray');
    end
    
    %% SVMѵ��   
    [w, b, class_name]  = li2nsvm_multiclass_lbfgs(X_train', trls', lambda);
    
    tic;
    [ttls_pred, ~]      = li2nsvm_multiclass_fwd(X_test', w, b, class_name);
    t=toc;
    test_time = t/length(ttls)
    
    reco_rate           = (sum(ttls_pred'==ttls))/length(ttls);
    
    disp(['Recognition rate on the ', dname, ' is ', num2str(roundn(reco_rate*100,-3)) '%']);
    reco_rates = [reco_rates; roundn(reco_rate*100,-3) ];
    
%     my_draw_cm;
%     mdl.P=M';mdl.B=D;
%     plot_dict(mdl);
    toc;
    
end

reco_rates



