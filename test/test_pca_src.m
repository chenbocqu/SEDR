

load USPS
dname       = 'USPS';

X_train     = normalize_mat(Train_DAT);
X_test      = normalize_mat(Test_DAT);

%% PCA ½µÎ¬

rdim = 196;

[M,eigvalue]    = PCA(X_train',rdim);
X_pca           = M'*X_train;

[M,eigvalue]    = PCA(X_test',rdim);
X_test          = M'*X_test;

%% SRC ÑµÁ·
[predictions,src_scores]    = src(X_pca',trainlabels,X_test',0.3);
reco_rate                   =  (sum(predictions==testlabels))/ length(testlabels);

disp(['Recognition rate on the ', dname, ' is ', num2str(roundn(reco_rate*100,-3)) '%']);