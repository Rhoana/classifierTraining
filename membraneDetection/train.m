function train()
maxNumberOfSamplesPerClass = 500;

mainData = get(1, 'UserData');
imgNames = mainData.imgNames;

fmPos = [];
fmNeg = [];

tic;
for i=1:length(imgNames),
  figure(i);
  clf;
  thisData = get(gcf, 'UserData');
  fg_image = thisData.Training_FG;
  bg_image = thisData.Training_BG;
  posPos = find(fg_image);
  posNeg = find(bg_image);
  if length(posPos)>0 | length(posNeg)>0
    name = imgNames(i).name
    load(strcat(name(1:end-10),'_fm.mat'));
    fm = reshape(fm,size(fm,1)*size(fm,2),size(fm,3));
    fm(isnan(fm))=0;
    fmPos = [fmPos; fm(posPos,:)];
    fmNeg = [fmNeg; fm(posNeg,:)];
    clear fm;
  end
end
toc;
clear posPos
clear posNeg

disp('training')
disp('Original number of samples per class: ');
disp('membrane: ');
disp(size(fmPos,1));
disp('not membrane: ');
disp(size(fmNeg,1));

if (size(fmPos,1) == 0) && (size(fmNeg,1) == 0),
   mainData.Forest = 0;
   set(1, 'UserData', mainData);
   return
end

tic;

y = [zeros(size(fmNeg,1),1);ones(size(fmPos,1),1)];
x = double([fmNeg;fmPos]);

extra_options.sampsize = [maxNumberOfSamplesPerClass, maxNumberOfSamplesPerClass];
forest = classRF_train(x, y, 300,5,extra_options);
%forest = classRF_train(x, y, 500,5);  
toc;

mainData.Forest = forest;
set(1, 'UserData', mainData);
