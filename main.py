from PIL import Image
import pytesseract
import queue

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
                pixdata[x, y] = 255
            else:
                pixdata[x, y] = 0
    return img
def depoint(img):
    """传入二值化后的图片进行降噪"""
    pixdata = img.load()
    w,h = img.size
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
                if max_x - min_x >  3:
                    # 宽度小于3的认为是噪点，根据需要修改
                    cuts.append((min_x,max_x))
    return cuts
def getCuts1(cuts):
    newcuts = []
    last = 0
    for item in cuts:
        if newcuts.__len__() == 0:
            newcuts.append(item)
            last = item[1]
        else:
            if last - item[0] > 10 or (newcuts[len(newcuts) - 1][1] - newcuts[len(newcuts) - 1][0] <= 30):
                first = newcuts[len(newcuts) - 1][0]
                last = max(last,item[1])
                newcuts.pop()
                # newcuts.remove(len(newcuts))
                newcuts.append((first, last))
            else:
                newcuts.append(item)
                last = item[1]
    return newcuts

def getCuts2 (img, cuts):
    '''分离粘连字符'''
    newcuts = []
    for item in cuts:
        if item[1] - item[0] > 60:
            mid = getCuts3(img, (item[0] + item[1]) // 2)
            newcuts.append((item[0], mid))
            newcuts.append((mid, item[1]))
        else:
            newcuts.append(item)
    return newcuts
def getCuts3(img, mid):
    offset = 20
    pixdata = img.load()
    w, h = img.size
    wl = 0 if(0 > mid - offset) else mid - offset
    wr = mid + offset if(mid + offset < w) else w
    min = h
    ret = 0
    for x in range(wl, wr):
        count = 0
        for y in range(h):
            if (pixdata[x, y] == 0):
                count = count + 1
        if min > count:
            min = count
            ret = x
    return ret

if __name__ == '__main__':
    src = None
    try:
        src = Image.open('first2.png')
    except FileNotFoundError:
        print('打开文件失败')
        exit(0)
    img = binarizing(src, 160)
    img = depoint(img)
    img.show()
    cuts = cfs(img)
    print(cuts)
    cuts = getCuts1(cuts)
    cuts = getCuts2(img, cuts)
    print(cuts)
    # images = []
    w, h = img.size
    for i, n in enumerate(cuts, 1):
        temp = img.crop((n[0], 0, n[1], h))  # 调用crop函数进行切割
        # images.append(temp)
        text = pytesseract.image_to_string(temp)
        print(text)
        temp.show()
    # text=pytesseract.image_to_string(img)
    # print(text)
    # img.show()
