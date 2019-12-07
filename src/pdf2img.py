from pdf2image import convert_from_path
import glob
import os

files = glob.glob('src/figures/*.pdf')
for item in files:
    
    file_name = item.split('\\')[1].split('.')[0]
    png_name = os.path.join(item.split('\\')[0] + '/png',file_name + '.png')
    pages = convert_from_path(item, 500)
    for page in pages:
        page.save(png_name, 'png')