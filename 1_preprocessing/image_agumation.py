from PIL import Image, ImageOps
import PIL.ImageOps

for j in range(3, 11):  
    for i in range(10):
        image = Image.open("./trainset21/s{}/img{}.jpg".format(j, i+11))
        rotated_image = image.rotate(10)
        rotated_image.save("./trainset21/s{}/img{}_rotate1.jpg".format(j, i+11))

    for i in range(10):
        image = Image.open("./trainset21/s{}/img{}.jpg".format(j, i+11))
        rotated_image = image.rotate(20)
        rotated_image.save("./trainset21/s{}/img{}_rotate2.jpg".format(j, i+11))

    for i in range(10):
        image = Image.open("./trainset21/s{}/img{}.jpg".format(j, i+11))
        rotated_image = image.rotate(30)
        rotated_image.save("./trainset21/s{}/img{}_rotate3.jpg".format(j, i+11))

    for i in range(10):
        image = Image.open("./trainset21/s{}/img{}.jpg".format(j, i+11))
        rotated_image = image.rotate(-10)
        rotated_image.save("./trainset21/s{}/img{}_rotate4.jpg".format(j, i+11))

    for i in range(10):
        image = Image.open("./trainset21/s{}/img{}.jpg".format(j, i+11))
        rotated_image = image.rotate(-20)
        rotated_image.save("./trainset21/s{}/img{}_rotate5.jpg".format(j, i+11))

    for i in range(10):
        image = Image.open("./trainset21/s{}/img{}.jpg".format(j, i+11))
        rotated_image = image.rotate(-30)
        rotated_image.save("./trainset21/s{}/img{}_rotate6.jpg".format(j, i+11))

    for i in range(10):
        image = Image.open("./trainset21/s{}/img{}.jpg".format(j, i+11))
        inverse_image = ImageOps.mirror(image)
        inverse_image.save("./trainset21/s{}/img{}_inverse.jpg".format(j, i+11))
    