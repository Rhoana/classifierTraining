function update_figures()
mainData = get(1, 'UserData');
imgNames = mainData.imgNames;
forest = mainData.Forest;

for i=1:length(imgNames),
  disp('preparation')
  tic;
  name = imgNames(i).name
  im = norm01(imread(name));
  load(strcat(name(1:end-10),'_fm.mat'));
  fm = reshape(fm,size(fm,1)*size(fm,2),size(fm,3));
  fm(isnan(fm))=0;
  clear fmNeg
  clear fmPos
  im=uint8Img(im(:,:,1));
  imsize = size(im);
  clear y
  clear im
  
  votes = zeros(imsize(1)*imsize(2),1);
  test = struct();
  toc;
  disp('prediction')
  tic;

% $$$   for j=1:4				% 
% $$$     [y_h,v] = classRF_predict(double(fm(j:4:end,:)), forest);
% $$$     votes(j:4:end,:)=v(:,2);    
% $$$ end
    
  if isa(forest, 'struct'),
    [y_h,v] = classRF_predict(double(fm), forest);
    votes = v(:,2);
    votes = double(votes)/max(votes(:));
  end
  votes = reshape(votes,imsize);
toc;
  disp('visualization')
  tic;
  im = norm01(imread(name));			% 
  size(im)
  %this illustration uses the thickened skeleton of the
  %segmentation
  %figure;
  %this is the skeletonized view
  figure(i); 
  curuserdata = get(i, 'UserData');
  if curuserdata.ShowOverlay,
    imshow(makeColorOverlay(votes,im));
  else
    imshow(im);
  end
  group = uibuttongroup('Position', [0.05, 0.025, 0.90, 0.05], 'BackgroundColor', 'red');
  uicontrol('Parent', group, 'Units', 'normalized', 'Position', [0, 0.05, 0.2, 0.90], 'String', 'Add FG', 'Callback', @draw_fg);
  uicontrol('Parent', group, 'Units', 'normalized', 'Position', [0.2, 0.05, 0.2, 0.90], 'String', 'Add BG', 'Callback', @draw_bg);
  uicontrol('Parent', group, 'Units', 'normalized', 'Position', [0.4, 0.05, 0.2, 0.9], 'String', 'Train', 'Callback', @retrain);  
  uicontrol('Parent', group, 'Units', 'normalized', 'Position', [0.6, 0.05, 0.2, 0.9], 'String', 'Save', 'Callback', @save_results);  
  uicontrol('Parent', group, 'Units', 'normalized', 'Position', [0.8, 0.05, 0.2, 0.9], 'String', 'Toggle', 'Callback', @toggle_overlay);  
  imwrite(makeColorOverlay(votes,im),strcat(name(1:6),'_overlay.tif'),'tif');
  %this is the thick membrane view
%  figure; imshow(makeColorOverlay(uint8Img(filterSmallRegions(votes>=0.5,1000)),uint8Img(im)));
  pause(1); %to give matlab time to show the figure
%  clear votes
%  clear y_hat
toc;
end
