      global preprocess; 
      %Normalize the data
      preprocess.Normalization = 1;
      %Evaluation Method: 0: Train-Test Split; 1: Cross Validation
      preprocess.Evaluation = 1;
      preprocess.root = '.';
      preprocess.output_file = sprintf('%s/_Result', preprocess.root);
      preprocess.input_file = sprintf('%s/example.data', preprocess.root);
      run = MIL_Run('DD -NumRuns 10 Aggregate avg');
