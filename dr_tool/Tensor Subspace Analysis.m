%X. He, D. Cai, and P. Niyogi, "Tensor Subspace Analysis", in NIPS'05, 2005. ( pdf )
%===========================================
fea_Train = fea(trainIdx,:); 
gnd_Train = gnd(trainIdx); 

options = []; 
options.Metric = 'Cosine'; 
options.NeighborMode = 'Supervised'; 
options.WeightMode = 'Cosine'; 
options.gnd = gnd_Train; 
W = constructW(fea_Train,options); 

options.nRepeat = 10;
fea_Train = reshape(fea_Train',32,32,size(fea_Train,1));
[U, V, eigvalue_U, eigvalue_V, posIdx] = TensorLPP(fea_Train, W, options);

[nSmp,nFea] = size(fea);
fea = reshape(fea',32,32,nSmp);
nRow = size(U,2);
nCol = size(V,2);
newfea = zeros(nRow,nCol,nSmp);
for i=1:nSmp
      newfea(:,:,i) = U'*fea(:,:,i)*V;
end
newfea = reshape(newfea,nRow*nCol,nSmp)';
newfea = newfea(:,posIdx);
%===========================================