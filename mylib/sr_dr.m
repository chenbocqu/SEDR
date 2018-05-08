function [P]=sr_dr(X,p,lambda1,lambda2,Max_iteration)

% output: P is a projection matrix of p by m,used to reduce the dimensionality of D
% input:  X is the dataset matrix of m by n
% this function's formula is argmin(P,alpha){norm(PX-PD*alpha,2)^2+lambda1*norm(alpha)+lambda2*norm(X-P'PX,2)^2}
% if you find it useful for your paper, please cite "Lei Zhang, Meng Yang, Zhizhao Feng and David Zhang. On the Dimensionality Reduction for Sparse Representation based Face Recognition. ICPR, Istanbul, Turkey,2010"

addpath L1Solvers;
n=size(X,2);
%% Initialize P,eg using PCA begin
numcomps=p;
meanvec = mean(X, 2);
meanarray = repmat(meanvec, 1, size(X,2));
A = X-meanarray;
covA = A*A';
[V, tem] = eig(covA);    %V eigenvector   D eigenvalue
P = V(:, (size(V, 2) - numcomps + 1) : size(V, 2));% numcomps eigenvalue minus correspondent eigenvectors
P = fliplr(P);
P=P';

%% Start iterating ...
Jnow=10; Jpre=0;iteration=0;

while  iteration<Max_iteration
    
    fprintf(['sr_dr: steperror=' num2str(Jnow-Jpre) ';and iteration=' num2str(iteration) '.\n']);
    
    %% Fix P, compute alpha, begin
    P_X=P*X;
    
    for i=1:n
        %     [s,status] = l1_ls(P*D, P_X(:,i), lambda1,1e-3,true);
        %     [s,status] = l1_ls(P*D, P_X(:,i), lambda1);
        P_D  =  [P_X(:,1:i-1) P_X(:,i+1:end)];
        D  =  [X(:,1:i-1) X(:,i+1:end)];
        [s, nIter] = SolveDALM(P_D, P_X(:,i), 'lambda',lambda1,'tolerance',1e-3);
        Y(:,i) = X(:,i) - D*s;
    end
    %Fix P,compute alpha,end
    
    %% Fix alpha,update P,begin
    % Y=X-D*alpha;
    Z=(Y*Y'-lambda2*X*X');
    [Vz,Dz] = eig(Z);           %Z*Vz = Vz*Dz. that is  inv(Vz)*Z*Vz=Dz;
    P = Vz(:,1:numcomps)';      % numcomps eigenvalue minus correspondent eigenvectors

%     Jnow=norm(P*X-P*D*alpha,2)^2+lambda1*norm(alpha,1)+lambda2*norm(X-P'*P*X,2)^2;
    iteration=iteration+1;
end

