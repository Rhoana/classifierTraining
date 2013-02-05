function imNorm = norm01(im)
  im = double(im);
  im = im(:,:,1);

  % invert and log transform, then reinvert
  im = 256 - im;
  im(im < 5) = 5;
  im = log(im);
  im = - im;

  im = im - min(im(:));
  m = max(im(:));
  if m ~= 0
    imNorm = im / m;
  else
    imNorm = im;
  end
  
