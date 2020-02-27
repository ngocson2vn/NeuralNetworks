# Convert pdf to png

## Install ghostscript
```
brew install ghostscript
```

## Convert
```
convert -density 300 notes.pdf -quality 100 -colorspace RGB png/notes.png
```

## Crop
```
mkdir cropped

for f in $(ls *.png); do convert -crop 2060x3500+210+10 $f cropped/$f; done

cd cropped/
convert -crop 3000x2682+0+155 notes-3.png notes-3-1.png
convert -crop 3000x3200+0+155 notes-4.png notes-4-1.png
convert -crop 3000x3200+0+155 notes-5.png notes-5-1.png
convert -crop 3000x3250+0+155 notes-6.png notes-6-1.png
convert -crop 3000x3200+0+155 notes-7.png notes-7-1.png
convert -crop 3000x3200+0+155 notes-8.png notes-8-1.png
convert -crop 3000x3200+0+155 notes-9.png notes-9-1.png
convert -crop 3000x3200+0+155 notes-10.png notes-10-1.png
convert -crop 3000x3250+0+155 notes-11.png notes-11-1.png
convert -crop 3000x2900+0+155 notes-12.png notes-12-1.png
```