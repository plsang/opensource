function [model_svm] = regular_svm_wrapper_linear(train_data, label, para)
% a regular svm wrapper using libsvm
% using binary labels -1/+1

% linear
%model_svm = train_linear(label, train_data, sprintf('-c %f -B -1 -q', para.C));
model_svm = train(label, train_data, sprintf('-s 0 -c %f -q ', para.C));
%model_svm = train(label, train_data, sprintf('-c %f -q', para.C));
model_svm.w = model_svm.w*model_svm.Label(1);
model_svm.w = model_svm.w';
model_svm.b = 0;
end