from PIL import Image
import pytesseract
import queue
from itertools import groupby
from PIL import Image
import uuid
import os
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

def denoise(img):
    """去除绿色干扰色"""
    pixdata = img.load()
    w, h = img.size
    for y in range(h):
        for x in range(w):
            r, g, b = pixdata[x, y]
            if r < 230 and g < 230 and b > 240 :
                img.putpixel((x, y), (0, 0, 255))
            else:
                img.putpixel((x, y), (255, 255, 255))
    return img
def binarizing(img,threshold):
    """传入image对象进行灰度、二值处理"""
    img = img.convert("L") # 转灰度
    # img.show()
    pixdata = img.load()
    w, h = img.size
    # 遍历所有像素，小于阈值的为黑色
    for y in range(h):
        for x in range(w):
            if pixdata[x, y] < threshold:
                pixdata[x, y] = 0
            else:
                pixdata[x, y] = 255
    return img
def depoint(img):
    """传入二值化后的图片进行降噪"""
    pixdata = img.load()
    w,h = img.size
    #左边界
    for y in range(1, h - 1):
        countl, countr = 0, 0
        #左边界
        if pixdata[0, y - 1] > 245:  # 上
            countl += 1
        if pixdata[1,y-1] > 245:
            countl += 1
        if pixdata[1,y] > 245:
            countl += 1
        if pixdata[1,y + 1] > 245:
            countl += 1
        if pixdata[0, y + 1] > 245:  # 下
            countl += 1
        if countl > 3:
            pixdata[0, y] = 255
        #右边界
        if pixdata[w - 1, y - 1] > 245:  # 上
            countr += 1
        if pixdata[w - 2, y - 1] > 245:
            countr += 1
        if pixdata[w - 2, y] > 245:
            countr += 1
        if pixdata[w - 2,y + 1] > 245:
            countr += 1
        if pixdata[w - 1, y + 1] > 245:  # 下
            countr += 1
        if countr > 3:
            pixdata[w - 1, y] = 255
    for y in range(1,h-1):
        for x in range(1,w-1):
            count = 0
            if pixdata[x,y-1] > 245:#上
                count = count + 1
            if pixdata[x,y+1] > 245:#下
                count = count + 1
            if pixdata[x-1,y] > 245:#左
                count = count + 1
            if pixdata[x+1,y] > 245:#右
                count = count + 1
            if pixdata[x-1,y-1] > 245:#左上
                count = count + 1
            if pixdata[x-1,y+1] > 245:#左下
                count = count + 1
            if pixdata[x+1,y-1] > 245:#右上
                count = count + 1
            if pixdata[x+1,y+1] > 245:#右下
                count = count + 1
            if count > 4:
                pixdata[x,y] = 255
    return img
# CFS连通域分割法
def cfs(img):
    """传入二值化后的图片进行连通域分割"""
    pixdata = img.load()
    w,h = img.size
    visited = set()
    q = queue.Queue()
    offset = [(-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1)]
    cuts = []
    for x in range(w):
        for y in range(h):
            x_axis = []
            #y_axis = []
            if pixdata[x,y] == 0 and (x,y) not in visited:
                q.put((x,y))
                visited.add((x,y))
            while not q.empty():
                x_p,y_p = q.get()
                for x_offset,y_offset in offset:
                    x_c,y_c = x_p+x_offset,y_p+y_offset
                    if (x_c,y_c) in visited:
                        continue
                    visited.add((x_c,y_c))
                    try:
                        if pixdata[x_c,y_c] == 0:
                            q.put((x_c,y_c))
                            x_axis.append(x_c)
                            #y_axis.append(y_c)
                    except:
                        pass
            if x_axis:
                min_x,max_x = min(x_axis),max(x_axis)
                if max_x - min_x > 2:
                    # 宽度小于3的认为是噪点，根据需要修改
                    cuts.append((min_x,max_x))
    return cuts

# def get_nearby_pix_value(img_pix,x,y,j):
#     """获取临近5个点像素数据"""
#     if j == 1:
#         return 0 if img_pix[x-1,y+1] == 0 else 1
#     elif j ==2:
#         return 0 if img_pix[x,y+1] == 0 else 1
#     elif j ==3:
#         return 0 if img_pix[x+1,y+1] == 0 else 1
#     elif j ==4:
#         return 0 if img_pix[x+1,y] == 0 else 1
#     elif j ==5:
#         return 0 if img_pix[x-1,y] == 0 else 1
#     else:
#         raise Exception("get_nearby_pix_value error")
def get_end_route(pix_img, w, height):
    """获取滴水路径"""
    default_char_w = 15  #默认字符宽度15
    default_drops_r = 1  #默认水滴半径1 即:水滴宽度 2B+1=3
    start_x = default_char_w# - 3
    # pix_img = img.load()
    # w, h = img.size
    # height = h
    left_limit = 0
    right_limit = start_x + 3
    end_route = []      #路径
    cur_p = (start_x,0) #当前水滴中心
    last_p = [start_x, 0]
    last_x = start_x
    end_route.append(cur_p)
    while cur_p[1] < (height-1): #触底结束
        sigma = 0
        max_w = 0
        next_x = cur_p[0]
        next_y = cur_p[1]
        # last_x = cur_p[0]
        #获取水滴周围5个区域的像素,求sigma
        #j = 4
        if (0 if 0 == pix_img[cur_p[0] + 2, cur_p[1]] else 1) > 0:
            sigma = max_w = 2
        #j = 5
        if (0 if 0 == pix_img[cur_p[0] - 2, cur_p[1]] else 1) > 0:
            sigma += 1
            if (0 == max_w):
                max_w = 1

        if pix_img[cur_p[0] - 2, cur_p[1] + 1] > 0 or pix_img[cur_p[0] - 1, cur_p[1] + 1] > 0 or pix_img[cur_p[0], cur_p[1] + 1] > 0:
            sigma += 5
            max_w = max(max_w, 5)
        if pix_img[cur_p[0] - 1, cur_p[1] + 1] > 0 or pix_img[cur_p[0], cur_p[1] + 1] > 0 or pix_img[cur_p[0] + 1, cur_p[1] + 1] > 0:
            sigma += 4
            max_w = max(max_w, 4)
        if pix_img[cur_p[0], cur_p[1] + 1] > 0 or pix_img[cur_p[0] + 1, cur_p[1] + 1] > 0 or pix_img[cur_p[0] + 2, cur_p[1] + 1] > 0:
            sigma += 3
            max_w = max(max_w, 3)

        # for i in range(1,6):
        #     cur_w = get_nearby_pix_value(pix_img,cur_p[0],cur_p[1],i) * (6-i)
        #     sum_n += cur_w
        #     if max_w < cur_w:
        #         max_w = cur_w
        if sigma == 0:
            # 如果全黑则看惯性
            max_w = 4
        if sigma == 15:
            max_w = 6
        #根据Wi的值确定方向
        if max_w == 1:
            next_x = cur_p[0] - 1    # (Xi-B-1, Xi+B-1)
            next_y = cur_p[1]
        elif max_w == 2:
            next_x = cur_p[0] + 1
            next_y = cur_p[1]
        elif max_w == 3:
            next_x = cur_p[0] + 1
            next_y = cur_p[1] + 1
        elif max_w == 5:
            next_x = cur_p[0] - 1
            next_y = cur_p[1] + 1
        elif max_w == 6:
            next_x = cur_p[0]
            next_y = cur_p[1] + 1
        elif max_w == 4:
            if sigma == 0 or last_x >= cur_p[0]:
                next_x = cur_p[0]
                next_y = cur_p[1] + 1
            elif last_x < cur_p[0]:
                next_x = cur_p[0] + 1
                next_y = cur_p[1] + 1
        else:
            raise Exception("get end route error")
        if next_x > cur_p[0]:
            last_x = cur_p[0]

        if next_x > right_limit:
            next_x = right_limit
            next_y = cur_p[1] + 1
        if next_x < left_limit:
            next_x = left_limit
            next_y = cur_p[1] + 1
        if next_x == last_p[0] and next_y == last_p[1]: #防止左右死循环
            next_x = cur_p[0]
            next_y = cur_p[1] + 1
        last_p = (cur_p[0], cur_p[1])
        cur_p = (next_x,next_y)
        end_route.append(cur_p)
    return end_route

def dealpicture():
    src = None
    try:
        src = Image.open('pic2/30.png')  # 9只有2个
    except FileNotFoundError:
        print('打开文件失败')
        exit(0)
    # img = denoise(src)
    # img.show()
    img = binarizing(src, 120)
    # img = depoint(img)
    img.save('train/src.png')
    cuts = cfs(img)  # 先根据连通域算法分割
    print(cuts)
    w, h = img.size
    idx = 0
    for i, n in enumerate(cuts, 1):
        temp = img.crop((n[0], 0, n[1], h))  # 调用crop函数进行切割
        width, height = temp.size
        needCut = width - 15 > 8
        if not needCut:
            depoint(temp).save('train/%s.png' % uuid.uuid1())
            idx += 1

        cutimg = temp
        while needCut:
            pix = cutimg.load()
            route = get_end_route(pix, width, height)
            # print(route)
            route_x = [max(list(i))[0] for _, i in groupby(route, lambda x: x[1])]  # 切割路径x坐标
            # print(route_x)
            _min = min(route_x)
            _max = max(route_x)
            _cl = _max  # 右侧字符左边界，下面用于去除空白
            for _y, _x in enumerate(route_x):
                if pix[_x + 1, _y] == 0 and _x < _cl:
                    _cl = _x
            print(_cl)
            print(width, _min, _max)
            needCut = width - _min - 15 > 5 or idx >= 5
            ########描绘路径########
            img_route = Image.new("RGB", (width, height), (255, 255, 255))
            route_red = img_route.load()
            for tx in range(width):
                for ty in range(height):
                    if pix[tx, ty] == 0:
                        route_red[tx, ty] = (0, 0, 0)
            for tt in route:
                if pix[tt[0], tt[1]] == 0:
                    route_red[tt[0], tt[1]] = (140, 0, 0)
                else:
                    route_red[tt[0], tt[1]] = (255, 0, 0)
            img_route.save('train/red%s%s.png' % (i, idx))
            ####################
            img_left = Image.new("L", (_max + 1, h), 255)
            pix_l = img_left.load()
            img_right = Image.new("L", (width - _min, h), 255)
            pix_r = img_right.load()
            for _y, _x in enumerate(route_x):
                for px in range(width):
                    if px <= _x:
                        pix_l[px, _y] = pix[px, _y]
                    else:
                        pix_r[px - _min, _y] = pix[px, _y]
            # cutimg.save('train/all%s.png' % idx)
            depoint(img_left).save('train/%s.png' % uuid.uuid1())
            idx += 1
            if img_right.size[0] < 10:
                break
            cutimg = img_right.crop((_cl - _min, 0, img_right.size[0], h))  # img_right 切除空白
            if not needCut or cutimg.size[0] < 20:
                depoint(cutimg).save('train/%s.png' % uuid.uuid1())
                idx += 1
                break
            width, height = cutimg.size
            idx += 1

def traindata():
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    for r, d, f in os.walk('train'):
        for each in f:
            imgname = os.path.join(r, each)
            print(imgname)
            img = Image.open(imgname)
            # img = img.resize((15, 30), Image.ANTIALIAS)
            # img.save(imgname)
            data = np.array(binarizing(img, 120))
            label = r.split('/')[-1]
            train_data.append(np.reshape(data, 15 * 30))  # 展开成1维数组
            train_label.append(label)
    for r, d, f in os.walk('pic2'):
        for each in f:
            imgname = os.path.join(r, each)
            img = Image.open(imgname)
            data = np.array(binarizing(img, 120))
            label = r.split('/')[-1]
            test_data.append(np.reshape(data, 15 * 30))
            test_label.append(label)
    knn = KNeighborsClassifier()
    knn.fit(train_data, train_label)
    k = knn.predict(test_data)
    print("knn score:", accuracy_score(test_label, k))
if __name__ == '__main__':
    dealpicture()
    # traindata()
        # for j in route:
        #     pix[j[0], j[1]] = 255
        # temp.save('train/%s.png' % i)
    # text=pytesseract.image_to_string(img)
    # print(text)
    # img.show()
