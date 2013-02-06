%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%GPU Implementation of Random Forest Classifier - Training
%v0.1
%Seymour Knowles-Barley
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Based on
%* mex interface to Andy Liaw et al.'s C code (used in R package randomForest)
%* Added by Abhishek Jaiantilal ( abhishek.jaiantilal@colorado.edu )
%* License: GPLv2
%* Version: 0.02
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%use like this:
%   forest = gpuTrainRF(X,Y,ntree,mtry,extra_options);
%
% X: data matrix
% Y: target values
% ntree: number of trees (default 500)
% mtry: number of predictors to use for each split
%      (default is floor(sqrt(size(X,2))))
% extra_options represent a structure containing various misc. options to
%      control the RF
%
% NOTE: In this version extra_options are all ignored except for extra_options.sampsize
% extra_options.sampsize =  Size(s) of sample to draw. For classification,
%      if sampsize is a vector of the length the number of strata, then sampling is stratified by strata,
%      and the elements of sampsize indicate the numbers to be
%      drawn from the strata.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function forest = gpuTrain (X,Y,ntree,mtry,extra_options)

%fprintf(1, 'gpuTrainRF - init\n');
%tic;

%Label classes
orig_labels = sort(unique(Y));
Y_new = Y;
nclass = length(orig_labels);
new_labels = 1:nclass;
for i=1:length(orig_labels)
    Y_new(Y==orig_labels(i))=new_labels(i);
end

Y = int32(Y_new);
clear Y_new;

%Set defaults
if ~exist('ntree','var') || ntree<=0
    ntree = 500;
end
if ~exist('mtry','var') || mtry<=0 || mtry>size(X,2)
    mtry = floor(sqrt(size(X,2)));
end

%Init options
if exist('extra_options','var')
    if isfield(extra_options,'sampsize'); sampsize = extra_options.sampsize; end
    if isfield(extra_options,'classwt'); classwt = extra_options.classwt; end
end

if ~exist('sampsize','var');
    nsamples = int32(ones(1,nclass) * size(X,1) / nclass);
    %setting for without replacement (not implemented)
    %nsamples = int32(ones(1,nclass) * 0.632 * size(X,1) / nclass);
else
    nsamples = int32(sampsize);
end

if ~exist('classwt','var')
    classwt = ones(1,nclass, 'single');
else
    classwt = single(classwt);
end

%Sanity check
[N D] = size(X);

if length(Y)~=N,
    error('Y size is not the same as X size');
end

%Random number seeds - change these if you want different results
seed = 0;
sequencestart = 0;

samplefrom = int32(zeros(1,nclass));
maxTreeSize = int32(2*sum(nsamples)+1);
nodeStopSize = int32(1);

for c = 1:nclass
    samplefrom(c) = sum(Y==c);
end

maxnsamples = max(nsamples);
classindex = -ones(max(samplefrom)*nclass, 1, 'int32');

cioffset = 0;
for c = 1:nclass
    classindex((1:samplefrom(c))+cioffset) = find(Y==c)'-1;
    cioffset = cioffset + samplefrom(c);
end

%toc;
%fprintf(1, 'gpuTrainRF - alloc\n');
%tic;

dev_bagspace = gpuArray(-ones([maxnsamples*nclass, ntree], 'int32'));
dev_tempbag = gpuArray(-ones([maxnsamples*nclass, ntree], 'int32'));

dev_treemap = gpuArray(zeros(maxTreeSize, ntree*2, 'int32'));
dev_nodestatus = gpuArray(zeros(maxTreeSize, ntree, 'int32'));
dev_xbestsplit = gpuArray(zeros(maxTreeSize, ntree, 'single'));
%dev_nbestsplit = gpuArray(zeros(maxTreeSize, ntree, 'int32'));
%dev_bestgini = gpuArray(zeros(maxTreeSize, ntree, 'single'));
dev_bestvar = gpuArray(zeros(maxTreeSize, ntree, 'int32'));
dev_nodeclass = gpuArray(zeros(maxTreeSize, ntree, 'int32'));
dev_ndbigtree = gpuArray(zeros(ntree, 1, 'int32'));
dev_nodestart = gpuArray(zeros(maxTreeSize, ntree, 'int32'));
dev_nodepop = gpuArray(zeros(maxTreeSize, ntree, 'int32'));
dev_classpop = gpuArray(zeros(maxTreeSize*nclass, ntree, 'int32'));
dev_classweights = gpuArray(classwt);
dev_weight_left = gpuArray(zeros(nclass, ntree, 'int32'));
dev_weight_right = gpuArray(zeros(nclass, ntree, 'int32'));
dev_dimtemp = gpuArray(zeros(D, ntree, 'int32'));

dev_baggedx = gpuArray(zeros(sum(nsamples)*D, ntree, 'single'));
dev_baggedclass = gpuArray(zeros(sum(nsamples), ntree, 'int32'));

threadsPerBlock = 32;
gridSizeX = ceil(double(ntree)/threadsPerBlock);

%Get kernel for prediction

k_train = parallel.gpu.CUDAKernel('train2.ptx', 'train2.cu');

k_train.ThreadBlockSize = threadsPerBlock;
k_train.GridSize = gridSizeX;

%toc;
%fprintf(1, 'gpuTrainRF - call\n');
%tic;

%GPU call
[dev_treemap, dev_nodestatus, dev_xbestsplit, ...
     ...%dev_nbestsplit, dev_bestgini, ...
     dev_bestvar, dev_nodeclass, dev_nbigtree ...
...%     dev_nodestart, dev_nodepop, ...
...%     dev_classpop, dev_classweights, ...
...%     dev_weight_left, dev_weight_right, ...
...%     dev_dimtemp, dev_bagspace, dev_tempbag ...
    ] = feval(k_train, gpuArray(single(X)), N, D, int32(nclass), ...
    gpuArray(int32(Y)), gpuArray(int32(classindex)), ...
    gpuArray(int32(nsamples)), gpuArray(int32(samplefrom)), ...
    int32(maxnsamples), seed, sequencestart, ...
    int32(ntree), int32(maxTreeSize), int32(mtry), int32(nodeStopSize), ...
    dev_treemap, dev_nodestatus, dev_xbestsplit, ...%dev_nbestsplit, dev_bestgini, ...
    dev_bestvar, dev_nodeclass, dev_ndbigtree, ...
    dev_nodestart, dev_nodepop, ...
    dev_classpop, dev_classweights, ...
    dev_weight_left, dev_weight_right, ...
    dev_dimtemp, dev_bagspace, dev_tempbag, dev_baggedx, dev_baggedclass);

%toc;
%fprintf(1, 'gpuTrainRF - gather\n');
%tic;

treemap = gather(dev_treemap);
nodestatus = gather(dev_nodestatus);
xbestsplit = gather(dev_xbestsplit);
bestvar = gather(dev_bestvar);
nodeclass = gather(dev_nodeclass);
nbigtree = gather(dev_nbigtree);

%bags = gather(dev_bagspace);
% nodestart = gather(dev_nodestart);
% nodepop = gather(dev_nodepop);
% classpop = gather(dev_classpop);
% classweights = gather(dev_classweights);
% weight_left = gather(dev_weight_left);
% weight_right = gather(dev_weight_right);
% dimtemp = gather(dev_dimtemp);
% tempbag = gather(dev_tempbag);
%

%Optional buffers - just for debugging
%nbestsplit = gather(dev_nbestsplit);
%bestgini = gather(dev_bestgini);

%Get output
forest.nrnodes = maxTreeSize;
forest.ntree = ntree;
forest.xbestsplit = xbestsplit;
forest.classwt = classwt;
forest.cutoff = single(ones(1,nclass)./nclass);
forest.treemap = treemap;
forest.nodestatus = nodestatus;
forest.nodeclass = nodeclass;
forest.bestvar = bestvar;
forest.ndbigtree = nbigtree;
forest.mtry = mtry;
forest.orig_labels=orig_labels;
forest.new_labels=new_labels;
forest.nclass = nclass;
forest.outcl = [];
forest.counttr = [];
forest.proximity = [];
forest.localImp = [];
forest.importance = [];
forest.importanceSD = [];
forest.errtr = [];
forest.inbag = [];
forest.votes = [];
forest.oob_times = [];

% try
%     clear obj.dev_treemap;
%     clear obj.dev_nodestatus;
%     clear obj.dev_xbestsplit;
%     clear obj.dev_bestvar;
%     clear obj.dev_nodeclass;
%     clear obj.k_predict;
%     reset(gpuDevice(1));
% catch err
%     fprintf(1, 'Error unbinding from GPU: %s\n', err.message);
% end

%toc;
%fprintf(1, 'gpuTrainRF - done\n');

end