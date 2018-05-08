function [ Y ] = oopdr( X,d,K,plotNumber )

%% SR����
para.p              = d;      % ��ά��ά��
para.K              = K;

para.lambda1        = 2e-4;     % lambda1 ||S||_1
para.lambda2        = 1.5;
para.MaxIters       = 25;

para.draw           = false;

SC_param.mode       = 2;
SC_param.lambda     = 0;
SC_param.lambda2    = para.lambda1;

para.sc_para        = SC_param;

%%
[ P,B,X_train,J ]     = pca_sr_dr( X,para );

Y = P*X;

if nargin > 3
    figure(3);
    subplot(3,3,plotNumber);
    imagesc(P);
    xlabel(['�� ',num2str(plotNumber),' ������']);
end

end

