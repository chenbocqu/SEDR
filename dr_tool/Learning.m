%D. Cai, X. He, Y. Hu, J. Han, and T. Huang, "Learning a Spatially Smooth Subspace for Face Recognition", in CVPR'07, 2007. ( pdf )
%===========================================
% fea_Train = fea(trainIdx,:); 
% gnd_Train = gnd(trainIdx); 

% options.KernelType = 'Gaussian';
% options.t = 1;
% fea = rand(1000,50);
% [eigvector,eigvalue] = KPCA(options,fea,25);
% feaTest = rand(100,50);
% Ktest = constructKernel(feaTest,fea,options);
% Y = Ktest*eigvector;

fea = rand(50,70);
gnd = [ones(10,1);ones(15,1)*2;ones(10,1)*3;ones(15,1)*4];
options = [];
options.Fisherface = 1;
[eigvector, eigvalue] = LDA(gnd, options, fea);
Y = fea*eigvector;


% load YaleB_DR_DAT;
% fea_Train = Test_DAT';
% gnd_Train = testlabels'; 

% fea = rand(50,70);
% gnd = [ones(10,1);ones(15,1)*2;ones(10,1)*3;ones(15,1)*4];
% options = [];
% options.k = 5;
% options.NeighborMode = 'Supervised';
% options.gnd = gnd;
% [eigvector, eigvalue] = NPE(options, fea);
% Y = fea*eigvector;


% options = []; 
% options.Metric = 'Cosine'; 
% options.NeighborMode = 'Supervised'; 
% options.WeightMode = 'Cosine'; 
% options.gnd = gnd_Train; 
% W = constructW(fea_Train,options); 
% 
% options.Regu = 1;
% options.ReguAlpha = 0.1;
% options.ReguType = 'Custom';
% load('TensorR_32x32.mat');
% options.regularizerR = regularizerR;
% 
% [eigvector, eigvalue] = LPP(W, options, fea_Train);
% [eigvector, eigvalue] = LDA(gnd_Train, options, fea_Train);
% ...
% 
% newfea = fea*eigvector;
