
%tic;

%profile on
train_MED12
%p = profile('info');
%profsave(p, 'profile/train')
%profile off

%profile on
test_MED12
%p = profile('info');
%profsave(p, 'profile/test')
%profile off

toc;
