function im = draw_into(im, sketch)
  % from http://www.mathworks.com/matlabcentral/answers/9531
  % Except...
  % Matlab repeats the path multiple times, or bits of it, etc.
  % Documentation on the internals of the freehand object are
  % basically nonexistent.  tl;dr Matlab sucks.
  %
  % Full path appears to be in the last child.
  ch = get(sketch, 'Children');
  XData = get(ch(length(ch)), 'XData');
  YData = get(ch(length(ch)), 'YData');
  for ii=1:(length(XData)-1)
    x1 = XData(ii);
    x2 = XData(ii+1);
    y1 = YData(ii);
    y2 = YData(ii+1);
    steps = max(abs(x1 - x2), abs(y1 - y2)) + 1;
    xl = round(linspace(x1, x2, steps));
    yl = round(linspace(y1, y2, steps));
    for jj=1:steps,
      im(yl(jj), xl(jj)) = 1;
    end
  end
