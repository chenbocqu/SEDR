function [ Y,Ps ] = SEDR_Process( X,d,K,color_vec )

%% 
% ��SEDR�㷨�м���̽��п��ӻ�

%% SR����
para.p              = d;      % ��ά��ά��
para.K              = K;

para.lambda1        = 2e-4;     % lambda1 ||S||_1
para.lambda2        = 1.5;
para.MaxIters       = 36;

para.draw           = false;
para.color_vec      = color_vec;
para.visulable      = true;

SC_param.mode       = 2;
SC_param.lambda     = 0;
SC_param.lambda2    = para.lambda1;

para.sc_para        = SC_param;

%%
[ P,B,X_train,J,Ps ]     = SEDR( X,para );

Y = P*X;

end

