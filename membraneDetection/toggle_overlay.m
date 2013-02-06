function toffle_overlay(hObject, eventdata)

% Draw buttons are nested in buttongroup
parentfig = get(get(hObject, 'Parent'), 'Parent');
userdata = get(parentfig, 'UserData');
userdata.ShowOverlay = ~ userdata.ShowOverlay
set(parentfig, 'UserData', userdata);

end
