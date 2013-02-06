function overlayImg = makeColorOverlay(votes,grayImage)
% image and votes should be float, 0 to 1
grayWeight = .8;
    overlayImg = uint8(zeros(size(votes,1),size(votes,2),3));
    skelImg = skeletonize(votes>=0.5);
    blend_to_full_red = grayImage + (1.0 - grayImage) .* single(votes);
    overlayImg(:,:,1) = uint8(255 * blend_to_full_red);
    overlayImg(:,:,2) = uint8(grayImage * 255);
    overlayImg(:,:,3) = uint8(grayImage * 255);
    
    green = overlayImg(:, :, 2);
    green(skelImg > 0) = 255;
    overlayImg(:, :, 2) = green;
  end
