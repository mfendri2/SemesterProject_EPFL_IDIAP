 function settingsPrettyFig(fSize)
if nargin==0, fSize=12; end 

hf = gcf;
allAxesInFigure = findall(hf,'type','axes');
hLegend = findobj(hf, 'Type', 'Legend');

set(hf,'color','w'); % set figure background to white

for ax = allAxesInFigure'
set(ax.Title,'Interpreter', 'latex', 'fontsize',fSize)
set(ax,'TickLabelInterpreter', 'latex', 'fontsize',fSize)
set(ax.XLabel,'Interpreter', 'latex', 'fontsize',fSize)
set(ax.YLabel,'Interpreter', 'latex', 'fontsize',fSize)
set(ax.ZLabel,'Interpreter', 'latex', 'fontsize',fSize)
if(isgraphics(hLegend))
    set(hLegend, 'Interpreter', 'latex','Location','best');
end

end