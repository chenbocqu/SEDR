function [ P,B,S,J ]=pca_sr_dr(X,para)

%% 直接返回低维字典 & 投影矩阵

% output: P is a projection matrix of p by m,used to reduce the dimensionality of D
% input:  X is the dataset matrix of m by n
% this function's formula is argmin(P,alpha){norm(PX-PD*alpha,2)^2+lambda1*norm(alpha)+lambda2*norm(X-P'PX,2)^2}

numcomps        = para.p;
lambda1         = para.lambda1;
lambda2         = para.lambda2;
Max_iteration   = para.MaxIters;
K               = para.K;           % 字典大小
SC_param        = para.sc_para;     % 稀疏编码参数

if (isfield(para,'draw'))
    draw = para.draw;
else
    draw = false;
end

X               = normalize_mat(X);

[m,n]           = size(X); % 样本总数

%% Initialize P using PCA
meanvec         = mean(X, 2);
meanarray       = repmat(meanvec, 1, size(X,2));
A               = X-meanarray;
covA            = A*A';         % 协方差矩阵
[V, tem]        = eig(covA);    %V eigenvector   D eigenvalue
P               = V(:, (size(V, 2) - numcomps + 1) : size(V, 2)); % numcomps eigenvalue minus correspondent eigenvectors
P               = fliplr(P);
P               = P';

%% 初始化字典 & DCT
B               = odctdict(numcomps,K);

% SC_param.mode   = 2;
% % SC_param.lambda = lambda1;
% SC_param.lambda2 = lambda1;
% SC_param.pos    = 'ture';

%% Start iterating ...
iteration   = 1;
J           = [];

while  iteration <= Max_iteration
    
    %% Fix P
    P_X         = P * X;
    
    % compute S
    S           = mexLasso( P_X,B,SC_param ); 
    
    % update B
    B           = update_B(P_X, S, 1);
    
    %% Fix B,S  update P
    P           = update_P( P, X, B, S,lambda2 );
 
    %% print lost J
    j           = norm(P*X-B*S,2)^2 + lambda1*norm(S,1) + lambda2*norm(X-P'*P*X,2)^2;
    J           = [J,j];

    fprintf(['sr_dr: J = ' num2str(j) '; iter = ' num2str(iteration) '.\n']);
    
    iteration=iteration+1;
    
end

if draw
   
   % 从第二步开始画迭代曲线
   plot     ( J(:,2:end),'bo-',...
                         'MarkerEdgeColor','b',...
                         'MarkerFaceColor','w',...
                         'MarkerSize',4);
                     
   xlabel   ( '迭代次数' );
   ylabel   ( '优化目标J' );
   
   grid on;
   
end

