function draw_bg(hObject, eventdata)
  % Draw buttons are nested in buttongroup
  parentfig = get(get(hObject, 'Parent'), 'Parent');
  userdata = get(parentfig, 'UserData');
  sketch = imfreehand('Closed', false);
  userdata.Training_BG = draw_into(userdata.Training_BG, sketch);
  set(parentfig, 'UserData', userdata);
  update_fig(parentfig);
  delete(sketch);
