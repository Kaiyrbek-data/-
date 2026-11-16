"""Rebuild the defense PDF from the reviewed slide renders.

The editable original is saved in Spaces; PDF pages are exact raster exports.
"""
from pathlib import Path
from reportlab.pdfgen import canvas
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'presentation'/'slides'
output=ROOT/'reports'/'Defense_slides.pdf'
pages=sorted(source.glob('slide-*.png'))
if len(pages)!=17: raise ValueError(f'Expected 17 reviewed slides, found {len(pages)}')
document=canvas.Canvas(str(output),pagesize=(960,540))
document.setTitle('Прогноз повторной госпитализации пациентов с диабетом')
document.setAuthor('Zhenissov Kaiyrbek; Bekarys')
for page in pages:
    document.drawImage(str(page),0,0,width=960,height=540)
    document.showPage()
document.save()
print(output)
