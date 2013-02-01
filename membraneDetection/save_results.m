function save_results(hObject, eventdata)
   mainData = get(1, 'UserData');
   imgNames = mainData.imgNames;
   forest = mainData.Forest;

  for i=1:length(imgNames)
    figure(i);
    thisData = get(gcf, 'UserData');
    fg_image = thisData.Training_FG;
    bg_image = thisData.Training_BG;
    name = imgNames(i).name
    im = cat(3, 255 * bg_image, 255 * fg_image, 0 * bg_image);
    imwrite(im,strcat(name(1:6),'_train.tif'),'tif');
  end
  save('forest.mat', 'forest');
