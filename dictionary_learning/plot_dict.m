function [ out ] = plot_dict( mdl )

P = mdl.P;
B = mdl.B;

%%
ImB         = displayPatches(B);      % 
ImPB         = displayPatches(P'*B);      % 

close ;
figure;
%% ��ά�ֵ���ع��ֵ�
subplot(121); imagesc(ImB);  colormap('gray');xlabel('��ά�ֵ�B');
subplot(122); imagesc(ImPB); colormap('gray');xlabel('�ع���ά�ֵ�P^TB');

% figure;
% imagesc(P);colormap('gray');xlabel('ͶӰ����');

end

