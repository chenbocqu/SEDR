%D. Cai, X. He, J. Han, and H. Zhang, "Orthogonal Laplacianfaces for Face Recognition", IEEE Transactions on Image Processing, vol. 15, no. 11, pp. 3608-3614, November, 2006. ( pdf ) 
%===========================================
fea_Train = fea(trainIdx,:); 
gnd_Train = gnd(trainIdx); 

options = []; 
options.Metric = 'Cosine'; 
options.NeighborMode = 'Supervised'; 
options.WeightMode = 'Cosine'; 
options.gnd = gnd_Train; 
W = constructW(fea_Train,options); 

options.Regu = 0;
options.PCARatio = 1;
bSuccess = 0;
while ~bSuccess
      [eigvector, eigvalue, bSuccess] = OLPP(W, options, fea_Train);
end

newfea = fea*eigvector;
%===========================================
