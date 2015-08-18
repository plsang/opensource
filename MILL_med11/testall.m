function test_all
    num_segs = power(2,[2:10])/4; % 4           8          16          32          64         128         256         512        1024 s
    num_segs = fliplr(num_segs);
    results = {};
    for ii = 1:length(num_segs),
        num_seg = num_segs(ii);
        fprintf('***** num_agg = %d \n', num_seg);
        cmd = sprintf('classify -t example.data -sf 0 -n 0 -- train_test_validate_med2012 -ss 1 -ee 25 -mneg Inf -fname idensetraj.hoghof.hardbow.cb4000 -fdim 4000 -nagg %d -- seg_SVM -Kernel 0', num_seg);
        run = MIL_Run(cmd);
        results{ii} = run.BagAccuMED; 
    end
    output_file = '/net/per610a/export/das11f/plsang/codes/opensource/MILL/testall_results/results.mat';
    save(output_file, 'results');
end

