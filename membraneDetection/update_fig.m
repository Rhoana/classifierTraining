function update_fig(fignum)
  figure(5 + fignum);
  clf;
  userdata = get(fignum, 'UserData');
  imshow(cat(3,255 * userdata.Training_BG, 255 * userdata.Training_FG, 0 * userdata.Training_BG));
