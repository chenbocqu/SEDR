function M = constructM(fea,options)
%	Usage:
%	M = constructM(fea,options)
% 
%	fea: Rows of vectors of data points. Each row is x_i
%   options: Struct value in Matlab. The fields in options that can be set:
%                  
%           NeighborMode -  Indicates how to construct the graph. Choices
%                           are: 
%                'KNN'            -  Put an edge between two nodes if and
%                                    only if they are among the k nearst
%                                    neighbors of each other. You are
%                                    required to provide the parameter k in
%                                    the options. [Default One]
%               'Supervised'      -  Two variations:
%                                       1. k=0, Put an edge between two nodes 
%                                          if and only if they belong to same class. 
%                                       2. k>0, The distance between two nodes 
%                                          in the same class will be smaller than 
%                                          two nodes have diff. labels 
%                                    You are required to provide the label
%                                    information gnd in the options.
%                                              
%            k         -   The number of neighbors.
%            gnd       -   The parameter needed under 'Supervised'
%                          NeighborMode.  Colunm vector of the label
%                          information for each data point.
%
%    Examples:
%
%       fea = rand(50,15);
%       options = [];
%       options.NeighborMode = 'KNN';
%       options.k = 5;
%       M = constructM(fea,options);
%       
%       
%       fea = rand(50,15);
%       gnd = [ones(10,1);ones(15,1)*2;ones(10,1)*3;ones(15,1)*4];
%       options = [];
%       options.NeighborMode = 'Supervised';
%       options.gnd = gnd;
%       options.k = 0;
%       M = constructM(fea,options);
%       
%       fea = rand(50,15);
%       gnd = [ones(10,1);ones(15,1)*2;ones(10,1)*3;ones(15,1)*4];
%       options = [];
%       options.NeighborMode = 'Supervised';
%       options.gnd = gnd;
%       options.k = 4;
%       M = constructM(fea,options);
%       
%Reference:
%
%   Xiaofei He, Deng Cai, Shuicheng Yan, and Hong-Jiang
%   Zhang, "Neighborhood Preserving Embedding", Tenth IEEE International 
%   Conference on Computer Vision (ICCV'2005), 2005
%    
%   Sam Roweis & Lawrence Saul. "Nonlinear dimensionality reduction by 
%   locally linear embedding", Science, v.290 no.5500 , Dec.22, 2000. 
%   pp.2323--2326.
%   
%   http://www.cs.toronto.edu/~roweis/lle/code.html
%       
%
%    Written by Deng Cai (dengcai@gmail.com), Jan/2005, Jan/2007
% 



[nSmp, nFea] = size(fea);

switch lower(options.NeighborMode)
    case {lower('KNN')}
    	
		aa = sum(fea.*fea,2);
		ab = fea*fea';
	    Distance = repmat(aa, 1, nSmp) + repmat(aa', nSmp, 1) - 2*ab;
	    Distance = max(Distance,Distance');
	    Distance = Distance - diag(diag(Distance));
	    
        if options.k <= 0
            error('k must be greater than 0!');
        end
    case {lower('Supervised')}
        if options.k > 0
        	
		    aa = sum(fea.*fea,2);
		    ab = fea*fea';
	        Distance = repmat(aa, 1, nSmp) + repmat(aa', nSmp, 1) - 2*ab;
	        Distance = max(Distance,Distance');
	        Distance = Distance - diag(diag(Distance));
    
            WLDA = ones(nSmp,nSmp);
            Label = unique(options.gnd);
            nLabel = length(Label);
            for idx=1:nLabel
                classIdx = find(options.gnd==Label(idx));
                WLDA(classIdx,classIdx) = 0;
            end
            Distance = Distance+WLDA*(max(max(Distance))+10);
        end
    otherwise
        error('NeighborMode does not exist!');
end


if options.k == 0
    W = zeros(nSmp,nSmp);
    for ii=1:nSmp
        idx = find(options.gnd==options.gnd(ii));
        idx(find(idx==ii)) = [];
        z = fea(idx,:)-repmat(fea(ii,:),length(idx),1); % shift ith pt to origin
        C = z*z';                                        % local covariance
        tW = C\ones(length(idx),1);                           % solve Cw=1
        tW = tW/sum(tW);                  % enforce sum(w)=1
        W(idx,ii) = tW;
    end
    M = (eye(size(W)) - W);
    M = M*M';
    M = max(M,M');
    M = sparse(M);
else
    [sorted,index] = sort(Distance,2);
    neighborhood = index(:,2:(1+options.k));

    W = zeros(options.k,nSmp);
    for ii=1:nSmp
        z = fea(neighborhood(ii,:),:)-repmat(fea(ii,:),options.k,1); % shift ith pt to origin
        C = z*z';                                        % local covariance
        W(:,ii) = C\ones(options.k,1);                           % solve Cw=1
        W(:,ii) = W(:,ii)/sum(W(:,ii));                  % enforce sum(w)=1
    end

    M = sparse(1:nSmp,1:nSmp,ones(1,nSmp),nSmp,nSmp,4*options.k*nSmp);
    for ii=1:nSmp
        w = W(:,ii);
        jj = neighborhood(ii,:)';
        M(ii,jj) = M(ii,jj) - w';
        M(jj,ii) = M(jj,ii) - w;
        M(jj,jj) = M(jj,jj) + w*w';
    end
    M = max(M,M');
    M = sparse(M);
end
