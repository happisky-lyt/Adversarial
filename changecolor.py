from PIL import Image

i = 1
j = 1
img1 = Image.open( 'E:/python/kaiti/train_patch/changecolor/image1/2_30_1.png')#读取系统的内照片
img2 = Image.open( 'E:/python/kaiti/train_patch/changecolor/image1/2_30_1.jpg')#读取系统的内照片
print (img1.size)#打印图片大小
print (img1.getpixel((4,4)))

width = img1.size[0]#长度
height = img1.size[1]#宽度
for i in range(0,width):#遍历所有长度的点
    for j in range(0,height):#遍历所有宽度的点
        # if i ==65 and j==81:
           
        data1 = (img1.getpixel((i,j)))#打印该图片的所有点
        data2 = (img2.getpixel((i,j)))#打印该图片的所有点
        #     # print(data)
        # print (data)#打印每个像素点的颜色RGBA的值(r,g,b,alpha)
        # print (data[0])#打印RGBA的r值
        if (abs(data[0]-56)<10 and abs(data[1] -56)<10 and abs(data[2]-255)<10):#RGBA的r值大于170，并且g值大于170,并且b值大于170
            img.putpixel((i,j),(255,56,56))#则这些像素点的颜色改成大红色
img = img.convert("RGB")#把图片强制转成RGB
img.save('E:/python/kaiti/train_patch/changecolor/image2/2_30_1.png')#保存修改像素点后的图片
