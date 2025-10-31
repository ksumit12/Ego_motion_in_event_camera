import cairosvg
import glob

for svgfile in glob.glob('*.svg'):
    pngfile = svgfile.replace('.svg', '.png')
    print(f'Converting {svgfile} -> {pngfile}')
    cairosvg.svg2png(url=svgfile, write_to=pngfile)

