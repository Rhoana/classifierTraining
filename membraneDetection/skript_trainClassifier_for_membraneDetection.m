%author: Verena Kaynig
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%precompute and save feature matrizes
  cs = 29;
  ms = 3;
  csHist = cs;
  hessianSigma = 4;
  
  imgNames = dir('*_image.tif');
  
  for i=1:length(imgNames)
    name = imgNames(i).name
    %only extract features if not already presaved
    if ~exist(strcat(name(1:6),'_fm.mat'));
      disp('extracting membrane features');
      disp(name);
      im = norm01((imresize(imread(imgNames(i).name),1)));
      fm  = membraneFeatures(im, cs, ms, csHist);
      save(strcat(name(1:6),'_fm.mat'),'fm');
      clear fm
      clear im
    end
    
    if ~exist(strcat(name(1:6),'_train.tif'));
      im = imread(name);
      im(:,:,2) = im;
      im(:,:,3) = im(:,:,1);
      imwrite(im,strcat(name(1:6),'_train.tif'),'tif');
    end
  end

% Preload training data  
trainimgNames = dir('*_train.tif');
for i=1:length(trainimgNames)
  figure(i);
  clf;
  name = trainimgNames(i).name;
  im = imread(name);
  fg_image = (im(:,:,2)==255 & im(:,:,1)==0);
  bg_image = (im(:,:,1)==255 & im(:,:,2)==0);
  set(gcf, 'UserData', struct('Training_FG', fg_image, ...
                              'Training_BG', bg_image, ...
                              'ShowOverlay', 1));
end
  

% Store 
maindata = get(1, 'UserData');
maindata.imgNames = imgNames;
set(1, 'UserData', maindata);

train;
update_figures;

%When the result is fine, save the random forest classifier
%with this command:

%save forest.mat forest
