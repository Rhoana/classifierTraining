function overlayImg = makeColorOverlay(votes,grayImage)
grayWeight = 200;
    overlayImg = uint8(zeros(size(votes,1),size(votes,2),3));
    skelImg = skeletonize(votes>=0.5);
    overlayImg(:,:,1) = uint8(norm01(grayImage)*grayWeight+norm01(exp(votes))*(255-grayWeight));
    overlayImg(:,:,2) = uint8(norm01(grayImage)*grayWeight);
    overlayImg(:,:,3) = uint8(norm01(grayImage)*grayWeight);
    
    green = overlayImg(:, :, 2);
    green(skelImg > 0) = max(grayImage(:));
    overlayImg(:, :, 2) = green;
  end
