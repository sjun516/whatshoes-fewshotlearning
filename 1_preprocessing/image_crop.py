from PIL import Image

j = 1
for i in range(10, 20) :
    img = Image.open('./tempimage2/s{0}/{1}.jpeg'.format(i // 10, j))
    img_cropped = img.crop((1008, 1008, 3024, 2016))
    img_cropped.save('./testset21/s{0}/img{1}.jpg'.format(i // 10, j))
    j += 1
    if j > 10 : j = 1