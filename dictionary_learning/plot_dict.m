function [ out ] = plot_dict( mdl )

P = mdl.P;
B = mdl.B;

%%
ImB         = displayPatches(B);      % 
ImPB         = displayPatches(P'*B);      % 

close ;
figure;
%% 低维字典和重构字典
subplot(121); imagesc(ImB);  colormap('gray');xlabel('低维字典B');
subplot(122); imagesc(ImPB); colormap('gray');xlabel('重构高维字典P^TB');

% figure;
% imagesc(P);colormap('gray');xlabel('投影矩阵');

end

