"""Scientific QA sheets containing every complete saved-prediction parent plot."""
from pathlib import Path
from PIL import Image,ImageDraw

root=Path(__file__).resolve().parent/'analysis'
files=sorted(root.glob('two_row_reach_*.png'))
if len(files)!=12:raise ValueError('All12 DEV parents required')
for start in range(0,12,3):
    canvas=Image.new('RGB',(2400,1560),'white');draw=ImageDraw.Draw(canvas)
    draw.text((20,12),'Composite108 / constant64 — all saved K4, three targets, best + last; no new forward',fill='black')
    for column,path in enumerate(files[start:start+3]):
        with Image.open(path) as source:
            source=source.convert('RGB');source.thumbnail((800,1520));canvas.paste(source,(800*column,35))
    canvas.save(root/('contact_sheet_%d.png'%(start//3+1)))
print('Created4 sheets, all12 original plots included.')
