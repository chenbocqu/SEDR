
num_in_class = [];

num_class   = length(unique(ttls));

if min(ttls) == 0 || min(ttls_pred) == 0
    ttls        = ttls+1;
    ttls_pred   = ttls_pred+1;
end

for ci = 1 : num_class
    
    num             = size(find(ttls ==ci),2);
    fprintf('class %d : %d\n', ci,num);
    num_in_class    = [ num_in_class ; num ];
    name_class {ci} = ['num_',num2str(ci-1)];

end

% cf = cfmatrix(ttls, ttls_pred',unique(ttls),1);
% draw_cm(cf,name_class,num_class);

[confusion_matrix]=compute_confusion_matrix(ttls_pred,num_in_class,name_class);

fprintf('\nInitialization Is Done!\n')

