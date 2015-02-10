function ap = calcMap(confs, labelsVid)
%confs = confs + 0.000001*rand(size(confs));
[confS IX] = sort(confs, 'descend');
labels = labelsVid(IX);

numOfRels = 0;
ap = 0;
for i = 1 : size(confS, 1)
   if labels(i) == 1
       numOfRels = numOfRels + 1;
       ap = ap + (numOfRels ./ i);
   end
end
ap = ap ./ numOfRels;

end

