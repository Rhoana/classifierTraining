function overlayImg = makeColorOverlay(votes,grayImage)
% image and votes should be float, 0 to 1
grayWeight = .9;
    overlayImg = uint8(zeros(size(votes,1),size(votes,2),3));
    skelImg = skeletonize(votes>=0.5);
    overlayImg(:,:,1) = uint8(255 * (grayImage*grayWeight+single(votes)*(1-grayWeight)));
    overlayImg(:,:,2) = uint8(grayImage * 255);
    overlayImg(:,:,3) = uint8(grayImage * 255);
    
    green = overlayImg(:, :, 2);
    green(skelImg > 0) = 255;
    overlayImg(:, :, 2) = green;
  end
