function P = update_P( X, B, S,lambda )

%% This code solves the following problem:
% 
%    minimize_P   ||X - B*S||^2 + lambda*||X-P'PX||^2
%    subject to   P'P = I

%% compute size
L = size(X,1);
N = size(X,2);
M = size(S, 1);
I = eye(L);

%% ËõÐ´
BSXt        = B*S*X';
XXt         = X*X';
trXXt       = sum(sum(X.^2));

%% 
dual_lambda = 10*abs(rand(L,1)); % any arbitrary initialization should be ok.
lb          = zeros(size(dual_lambda));

options     = optimset('Algorithm','trust-region-reflective' ,'GradObj','on', 'Hessian','on','Display', 'on');
[x, fval, exitflag, output] = fmincon(@(x) fobj_update_P( x, XXt, BSXt,lambda ), dual_lambda, [], [], [], [], lb, [], [], options);

% output.iterations
fval_opt        = -0.5*N*fval;
dual_lambda     = x;

Pt = ( (1-lambda)*XXt + diag(dual_lambda) ) \ BSXt';
P= Pt';

fobjective = fval_opt;

return;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [f,g,H] = fobj_update_P( dual_lambda, XXt, BSXt, lambda )
% Compute the objective function value at x
L   = size(BSXt,1);
M   = length(dual_lambda);

XXt_inv = inv( (1-lambda)*XXt + diag(dual_lambda) );

% trXXt = sum(sum(X.^2));
if L>M
    % (M*M)*((M*L)*(L*M)) => MLM + MMM = O(M^2(M+L))
    f = - trace(XXt_inv*(BSXt'*BSXt)) - sum(dual_lambda);
    
else
    % (L*M)*(M*M)*(M*L) => LMM + LML = O(LM(M+L))
    f = - trace(BSXt*XXt_inv*BSXt')  - sum(dual_lambda);
end
f= -f;

if nargout > 1   % fun called with two output arguments
    % Gradient of the function evaluated at x
    g       = zeros(M,1);
    temp    = BSXt*XXt_inv;
    g       = sum(temp.^2) - 1;
    g       = -g;
    
    if nargout > 2
        % Hessian evaluated at x
        % H = -2.*((SSt_inv*XSt'*XSt*SSt_inv).*SSt_inv);
        H   = -2.*((temp'*temp).*XXt_inv);
        H   = -H;
    end
end

return