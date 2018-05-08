
clear;
load Fault_Original;

ratio = 0.75;
len   = size( cnnData,1 );

randId      = randperm( len );

%% 随机打乱
cnnData     = cnnData(randId,:);
cnnLabel    = cnnLable(randId,:);

idFloor     = floor(len *ratio); % 

Train_DAT    = cnnData( 1:idFloor, : )';
trainlabels  = cnnLabel(1:idFloor, :)';

Test_DAT    = cnnData( idFloor+1:end, : )';
testlabels  = cnnLabel(idFloor+1:end, :)';

%% 下采样
% Train_DAT   = resample(Train_DAT,2000,20000);
% Test_DAT   = resample(Test_DAT,2000,20000);

save('Fault_DATA', 'Train_DAT', 'trainlabels','Test_DAT','testlabels');





