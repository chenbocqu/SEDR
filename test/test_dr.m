
% load AR_DR_DAT
% load YaleB_DR_DAT
load USPS

%% ����
X                   = Train_DAT;
para.p              = 196;      % ��ά��ά��
para.lambda1        = 0.03;
para.lambda2        = 1.5;
para.MaxIters       = 20;
para.K              = 225;
para.draw           = true;

SC_param.mode   = 2;
SC_param.lambda = para.lambda1;
% SC_param.pos    = 'ture';

%% ��ά
% [ P,B,J ]           = my_sr_dr( X,para );
[ P,B,S,J ]         = pca_sr_dr( X,para );

%% SVM ѵ��
mdl                 = train(trainlabels', S' );

Test_DAT            = normalize_mat(Test_DAT);
%% ����
S_test              = mexLasso( P*Test_DAT,B,SC_param ); 
[predicted_label, accuracy, dp] = predict(testlabels', S_test', mdl );

%% ���ӻ��ֵ�
figure(2);ImD=displayPatches(B);      % ���ӻ��ֵ�
figure(2); imagesc(ImD); colormap('gray');

