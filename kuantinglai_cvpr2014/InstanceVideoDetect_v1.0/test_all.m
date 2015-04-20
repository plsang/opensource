function test_all
    num_segs = power(2,[5:10])/4; % 4           8          16          32          64         128         256         512        1024 s
    results = {};
    for ii = 1:length(num_segs),
        num_seg = num_segs(ii);
        fprintf('***** num_agg = %d \n', num_seg);
        results{ii} = main_med('idensetraj.hoghof.hardbow.cb4000', 4000, 'r4', Inf, num_seg);
    end
    output_file = '/net/per610a/export/das11f/plsang/codes/opensource/kuantinglai_cvpr2014/InstanceVideoDetect_v1.0/testall_results/results.mat';
    save(output_file, 'results');
end
