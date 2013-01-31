function draw_fg(hObject, eventdata)
  % Draw buttons are nested in buttongroup
  parentfig = get(get(hObject, 'Parent'), 'Parent');
  userdata = get(parentfig, 'UserData');
  sketch = imfreehand('Closed', false);
  userdata.Training_FG = draw_into(userdata.Training_FG, sketch);
  set(parentfig, 'UserData', userdata);
  update_fig(parentfig);
  delete(sketch);
