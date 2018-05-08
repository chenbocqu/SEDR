
%% 验证稀疏嵌入SE

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

isdr            = true; % 是否降维
issr            = true; % 是否用字典学习

%% SVM学习率
lambda          = 0.03;

%% SR参数
para.K              = 100;

para.lambda1        = 3e-2;     % lambda1 ||S||_1
para.lambda2        = 0.7;
para.MaxIters       = 25;

para.draw           = true;

SC_param.mode       = 2;
SC_param.lambda     = para.lambda1;
% SC_param.lambda2    = para.lambda1;

para.sc_para        = SC_param;

reco_rates= [];
% 61.23

Ps = [];
Bs = [];

cnt = 0;

range = [5:5:30];

for r=range
    
    cnt     = cnt + 1;
    
    disp(['running ', num2str(cnt), ' th time ... ']);
    
    load USPS
    dname = 'USPS dataset';
    para.K              = 196;
    
    tr_dat  = Train_DAT;
    tt_dat  = Test_DAT;
    trls    = trainlabels;
    ttls    = testlabels;
    
    clear Train_DAT Test_DAT trainlabels testlabels;
    
    %% 参数设置
    rdim            = r ;       % i^2;      % 降维维度 PCA，SRDR[ 25 -> 169 ]
    para.p          = rdim;     % 降维的维数
    
    
    %% 数据归一化
    X_train         = normalize_mat(tr_dat);
    X_test          = normalize_mat(tt_dat);
    
    [ P,B,X_train,J ]     = pca_sr_dr( X_train,para );
    
    %% 保存
    Ps{cnt}      = P;
    Bs{cnt}      = B;
    
    X_test          = mexLasso( P*X_test,B,SC_param );
    
    %% SVM训练
    [w, b, class_name]  = li2nsvm_multiclass_lbfgs(X_train', trls', lambda);
    
    [ttls_pred, ~]      = li2nsvm_multiclass_fwd(X_test', w, b, class_name);
    reco_rate           = (sum(ttls_pred'==ttls))/length(ttls);
    
    disp(['Recognition rate on the ', dname, ' is ', num2str(roundn(reco_rate*100,-3)) '%']);
    reco_rates = [reco_rates; roundn(reco_rate*100,-3) ];
    
    %% 绘制低维字典和重构字典
    %     mdl.P=P;mdl.B=B;
    %     plot_dict(mdl);
end

%% 画图验证稀疏嵌入
r=range;
figure;
for i = 1:cnt
    
    P = Ps{i};
    B = Bs{i};
    
    figure;
    ImPB         = displayPatches(P'*B);      %   
    close ;
    
    rdim = r(i);%r(i)^2;
    
    %% 低维字典和重构字典
    subplot(3,2,i); imagesc(ImPB);  colormap('gray');
    xlabel(['r = ',num2str(rdim)]);
    
end

plot(r,reco_rates/100,'bo-.'); grid on;
xlabel('维度（r）');ylabel('精度（%）');
legend('OOPDR');

% 从第二步开始画迭代曲线
plot     ( J(:,2:end),'bo-',...
            'MarkerEdgeColor','b',...
            'MarkerFaceColor','w',...
            'MarkerSize',4);

xlabel   ( '迭代次数' );
ylabel   ( '优化目标J' );

grid on;

reco_rates

% OOPDR + SVM：93.622%
% PCA+SR+SVM：94.569%（ r=196, K=350 ）



