%feature parameters
nori  = 9;
border= 2;
stridew=16;
strideh=16;
scaleratio=sqrt(sqrt(sqrt(2)));
levels = 4;

num_sample = 20;
rand('seed',0);
im = 'neg_test.png';
tic;
  feats_single=compute_features(im,border,nori,num_sample,'a');
toc;

im = 'pos_test.png';
if(0)
  tic;
    [feats,win_posw,win_posh,winw,winh] = ...
        compute_features_scale_space(im,border,scaleratio,nori,stridew,strideh,'a');
  toc;
end


