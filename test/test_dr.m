
% load AR_DR_DAT
% load YaleB_DR_DAT
load USPS

%% 参数
X                   = Train_DAT;
para.p              = 196;      % 降维的维数
para.lambda1        = 0.03;
para.lambda2        = 1.5;
para.MaxIters       = 20;
para.K              = 225;
para.draw           = true;

SC_param.mode   = 2;
SC_param.lambda = para.lambda1;
% SC_param.pos    = 'ture';

%% 降维
% [ P,B,J ]           = my_sr_dr( X,para );
[ P,B,S,J ]         = pca_sr_dr( X,para );

%% SVM 训练
mdl                 = train(trainlabels', S' );

Test_DAT            = normalize_mat(Test_DAT);
%% 测试
S_test              = mexLasso( P*Test_DAT,B,SC_param ); 
[predicted_label, accuracy, dp] = predict(testlabels', S_test', mdl );

%% 可视化字典
figure(2);ImD=displayPatches(B);      % 可视化字典
figure(2); imagesc(ImD); colormap('gray');

