function [ Y,Ps ] = SEDR_Process( X,d,K,color_vec )

%% 
% 对SEDR算法中间过程进行可视化

%% SR参数
para.p              = d;      % 降维的维数
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

