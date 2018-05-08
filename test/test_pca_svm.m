
% warning off;
load USPS
dname = 'USPS';

X_train     = normalize_mat(Train_DAT);
X_test      = normalize_mat(Test_DAT);

%% PCA ½µÎ¬
rdim = 169;

M           = Eigenface_f(X_train,rdim); % P = M'

X_train             = M'*X_train;
X_test              = M'*X_test;

% Use a linear support vector machine classifier
lambda          = 2e-6;

[w, b, class_name] = li2nsvm_multiclass_lbfgs(X_train', trainlabels', lambda);

[ttls_pred, ~]  = li2nsvm_multiclass_fwd(X_test', w, b, class_name);
reco_rate       =  (sum(ttls_pred'==testlabels))/length(testlabels);

disp(['Recognition rate on the ', dname, ' is ', num2str(roundn(reco_rate*100,-3)) '%']);
