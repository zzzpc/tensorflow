import  numpy as np
import  random
import  math
cc= [ "00000000"  ,   "00000001"  ,   "00000010"  ,   "00000011"  ,   "00000100"  ,   "00000101"  ,   "00000110"  ,   "00000111"  ,   "00001000"  ,   "00001001"  ,   "00001010"
,   "00001011"  ,   "00001100"  ,   "00001101"  ,   "00001110"  ,   "00001111"  ,   "00010000"  ,   "00010001"  ,   "00010010"  ,   "00010011"  ,   "00010100"  ,   "00010101"  
,   "00010110"  ,   "00010111"  ,   "00011000"  ,   "00011001"  ,   "00011010"  ,   "00011011"  ,   "00011100"  ,   "00011101"  ,   "00011110"  ,   "00011111"  ,   "00100000"  
,   "00100001"  ,   "00100010"  ,   "00100011"  ,   "00100100"  ,   "00100101"  ,   "00100110"  ,   "00100111"  ,   "00101000"  ,   "00101001"  ,   "00101010"  ,   "00101011"  
,   "00101100"  ,   "00101101"  ,   "00101110"  ,   "00101111"  ,   "00110000"  ,   "00110001"  ,   "00110010"  ,   "00110011"  ,   "00110100"  ,   "00110101"  ,   "00110110"  
,   "00110111"  ,   "00111000"  ,   "00111001"  ,   "00111010"  ,   "00111011"  ,   "00111100"  ,   "00111101"  ,   "00111110"  ,   "00111111"  ,   "01000000"  ,   "01000001"  
,   "01000010"  ,   "01000011"  ,   "01000100"  ,   "01000101"  ,   "01000110"  ,   "01000111"  ,   "01001000"  ,   "01001001"  ,   "01001010"  ,   "01001011"  ,   "01001100"  
,   "01001101"  ,   "01001110"  ,   "01001111"  ,   "01010000"  ,   "01010001"  ,   "01010010"  ,   "01010011"  ,   "01010100"  ,   "01010101"  ,   "01010110"  ,   "01010111"  
,   "01011000"  ,   "01011001"  ,   "01011010"  ,   "01011011"  ,   "01011100"  ,   "01011101"  ,   "01011110"  ,   "01011111"  ,   "01100000"  ,   "01100001"  ,   "01100010"  
,   "01100011"  ,   "01100100"  ,   "01100101"  ,   "01100110"  ,   "01100111"  ,   "01101000"  ,   "01101001"  ,   "01101010"  ,   "01101011"  ,   "01101100"  ,   "01101101"  
,   "01101110"  ,   "01101111"  ,   "01110000"  ,   "01110001"  ,   "01110010"  ,   "01110011"  ,   "01110100"  ,   "01110101"  ,   "01110110"  ,   "01110111"  ,   "01111000"  
,   "01111001"  ,   "01111010"  ,   "01111011"  ,   "01111100"  ,   "01111101"  ,   "01111110"  ,   "01111111"  ,   "10000000"  ,   "10000001"  ,   "10000010"  ,   "10000011"  
,   "10000100"  ,   "10000101"  ,   "10000110"  ,   "10000111"  ,   "10001000"  ,   "10001001"  ,   "10001010"  ,   "10001011"  ,   "10001100"  ,   "10001101"  ,   "10001110"  
,   "10001111"  ,   "10010000"  ,   "10010001"  ,   "10010010"  ,   "10010011"  ,   "10010100"  ,   "10010101"  ,   "10010110"  ,   "10010111"  ,   "10011000"  ,   "10011001"  
,   "10011010"  ,   "10011011"  ,   "10011100"  ,   "10011101"  ,   "10011110"  ,   "10011111"  ,   "10100000"  ,   "10100001"  ,   "10100010"  ,   "10100011"  ,   "10100100"  
,   "10100101"  ,   "10100110"  ,   "10100111"  ,   "10101000"  ,   "10101001"  ,   "10101010"  ,   "10101011"  ,   "10101100"  ,   "10101101"  ,   "10101110"  ,   "10101111"  
,   "10110000"  ,   "10110001"  ,   "10110010"  ,   "10110011"  ,   "10110100"  ,   "10110101"  ,   "10110110"  ,   "10110111"  ,   "10111000"  ,   "10111001"  ,   "10111010"  
,   "10111011"  ,   "10111100"  ,   "10111101"  ,   "10111110"  ,   "10111111"  ,   "11000000"  ,   "11000001"  ,   "11000010"  ,   "11000011"  ,   "11000100"  ,   "11000101"  
,   "11000110"  ,   "11000111"  ,   "11001000"  ,   "11001001"  ,   "11001010"  ,   "11001011"  ,   "11001100"  ,   "11001101"  ,   "11001110"  ,   "11001111"  ,   "11010000"  
,   "11010001"  ,   "11010010"  ,   "11010011"  ,   "11010100"  ,   "11010101"  ,   "11010110"  ,   "11010111"  ,   "11011000"  ,   "11011001"  ,   "11011010"  ,   "11011011"  
,   "11011100"  ,   "11011101"  ,   "11011110"  ,   "11011111"  ,   "11100000"  ,   "11100001"  ,   "11100010"  ,   "11100011"  ,   "11100100"  ,   "11100101"  ,   "11100110"  
,   "11100111"  ,   "11101000"  ,   "11101001"  ,   "11101010"  ,   "11101011"  ,   "11101100"  ,   "11101101"  ,   "11101110"  ,   "11101111"  ,   "11110000"  ,   "11110001"  
,   "11110010"  ,   "11110011"  ,   "11110100"  ,   "11110101"  ,   "11110110"  ,   "11110111"  ,   "11111000"  ,   "11111001"  ,   "11111010"  ,   "11111011"  ,   "11111100"  
,   "11111101"  ,   "11111110"  ,   "11111111"]

r=6.5
cn=0
class Ent(object):
    def __init__(self):
        self.pos=[0]*3
        self.c=['0']*3

NDB_record=[]
for i in range(60000):
    item=Ent()
    NDB_record.append(item)
NDB_record[0].pos[1]=5


p=[0]*4
q=[0]*9
posbility=[[0]*2 for i in range(9)]
Pos=[[0]*2 for i in range(9000)]


def diff(i):
    Ndiff=0
    Nsame=0
    for j in range(1,4):
        Ndiff= Ndiff+ j * p[j] * q[i]

    for j in range(1,4):
        Nsame= Nsame+ (3-j) * p[j]/9

    pdiff=0
    pdiff=Ndiff/(Ndiff+Nsame)
    return pdiff

def init(s):

    p[0] = 0
    p[1] = 0.70
    p[2] = 0.24
    p[3] = 1 - p[1] - p[2]
    cn = int(len(s) * r + 0.5)

    q[0] = 0
    q[1] = 0.93
    q[2] = 0.01
    q[3] = 0.01
    q[4] = 0.01
    q[5] = 0.01
    q[6] = 0.01
    q[7] = 0.01
    q[8] = 0.01

    for i in range(9):
        posbility[i][0]=diff(i)
        posbility[i][1]=1-posbility[i][0]


def genRint(n):
    return np.random.randint(0,n)

def genRflo():
    return random.random()

def generateBit(l):
    if l<q[0] : return 0
    if l < q[0] + q[1]: return 1
    if l < q[0] + q[1] + q[2]: return 2
    if l < q[0] + q[1] + q[2] + q[3]: return 3
    if l < q[0] + q[1] + q[2] + q[3] + q[4]: return 4
    if l < q[0] + q[1] + q[2] + q[3] + q[4] + q[5]: return 5
    if l < q[0] + q[1] + q[2] + q[3] + q[4] + q[5] + q[6]: return 6
    if l < q[0] + q[1] + q[2] + q[3] + q[4] + q[5] + q[6]+ q[7]: return 7
    else : return 8

def addToNDB(Ent):
    global  cn
    for i in range(3):
        NDB_record[cn].pos[i] = Ent.pos[i]
        NDB_record[cn].c[i] = Ent.c[i]


def f(s):
    global r,cn

    item=Ent()
    i=0
    len_s = len(s)
    print("长度为",len_s)
    n=int(len_s*r+0.5)

    while cn<n:
        t=genRflo()
        if t<p[1]:
            diff1=generateBit(genRflo())
            attr1=genRint(len_s/9)
            item.pos[0]=diff1+attr1*9
            item.c[0]=chr(ord('1')+ord('0')-ord(s[item.pos[0]]))

            same1=genRint(9)
            attr2=genRint(len_s/9)
            while same1+ attr2 *9 == item.pos[0]:
                same1=genRint(9)
            item.pos[1]=same1+ attr2 *9
            item.c[1]=s[item.pos[1]]

            same2 = genRint(9)
            attr3 = genRint(len_s / 9)
            while same2 + attr3 * 9 == item.pos[0] | same2 + attr3 * 9 == item.pos[1]:
                same2 = genRint(9)
            item.pos[2] = same2 + attr3 * 9
            item.c[2] = s[item.pos[2]]

        if t<p[1] + p[2]:
            diff1 = generateBit(genRflo())
            attr1 = genRint(len_s / 9)
            item.pos[0]= diff1 + attr1 * 9
            item.c[0] = chr(ord('1')+ord('0')-ord(s[item.pos[0]]))

            diff2 = generateBit(genRflo())
            attr2 = genRint( len_s/ 9)
            while diff2+ attr2 *9 == item.pos[0]:
                diff2 = generateBit(genRflo())
            item.pos[1]=diff2+ attr2 *9
            item.c[1]=chr(ord('1')+ord('0')-ord(s[item.pos[1]]))

            same1 = genRint(9)
            attr3 = genRint(len_s / 9)
            while same1 + attr3 * 9 == item.pos[0] | same1 + attr3 * 9 == item.pos[1]:
                same1 = genRint(9)
            item.pos[2]= same1 + attr3 * 9
            item.c[2] = s[item.pos[2]]

        else:
            diff1 = generateBit(genRflo())
            attr1 = genRint(len_s / 9)
            item.pos[0] = diff1 + attr1 * 9
            item.c[0] = chr(ord('1')+ord('0')-ord(s[item.pos[0]]))

            diff2 = generateBit(genRflo())
            attr2 = genRint(len_s / 9)
            while diff2 + attr2 * 9 == item.pos[0]:
                diff2 = generateBit(genRflo())
            item.pos[1] = diff2 + attr2 * 9
            item.c[1] = chr(ord('1')+ord('0')-ord(s[item.pos[1]]))


            diff3 = generateBit(genRflo())
            attr3 = genRint(len_s / 9)
            while diff3 + attr3 * 9 == item.pos[0] | diff3 + attr3 * 9 == item.pos[1]:
                diff3 = generateBit(genRflo())
            item.pos[2] = diff3 + attr3 * 9
            item.c[2] = chr(ord('1')+ord('0')-ord(s[item.pos[2]]))
        addToNDB(item)
        cn+=1

    for i in range(cn):
        for j in range(3):

            if NDB_record[i].c[j] == '0' :

                Pos[NDB_record[i].pos[j]][0]+=1
            else:

                Pos[NDB_record[i].pos[j]][1]+=1

def print_Ndb(s):
    print(len(s)/9)
    for i in range(int(len(s)/9)):
        print(str(Pos[i][0])+" "+str(Pos[i][1]))


def pr(index , num):
    tmp=index%9
    pdiff=posbility[tmp][0]
    psame=posbility[tmp][1]

    if pdiff==0:
        if num==0:
            if Pos[index][0] == 0 :
                return 1
            else:
                return 0
        else:
            if Pos[index][1] == 0:
                return 1
            else:
                return 0

    var1 = Pos[index][1] * math.log(pdiff) + Pos[index][0] * math.log(psame)
    var2 = Pos[index][0] * math.log(pdiff) + Pos[index][1] * math.log(psame)

    denominator=math.exp(var1)+math.exp(var2)
    if denominator==0:
        if Pos[index][0]<Pos[index][1]:
            if num==0: return 0
            else:      return 1

        else:
            if num == 1:
                return 0
            else:
                return 1

    #print(denominator)
    if num==1:
        pr1=var2-math.log(denominator)
        return pr1
    else:
        pr0=var1-math.log(denominator)
        return pr0

def calculate(index, real):
    sum=0
    index+=1
    for i in range(8):
        tmp=pr(index+i,ord(cc[real][i])-ord('0'))
        if tmp==1: return 0
        sum+=tmp

    sum+=math.log(real)
    return math.exp(sum)/255

def E(index,s):
    tmp=0
    for i in range(1,256):
        tmp+=calculate(index, i)


    if s[index] == '0':        tmp =-tmp  #需改进

    return tmp

def readfile(s):
    num=0
    Gen = [0] * int(len(s)/9)
    for i in range(len(s)):
        if i%9==0:
            Gen[num]=E(i,s)
            num+=1
    return Gen


def main(s):
    init(s)   #初始化参数设置
    f(s)     #生成对应串的负数据库以及统计过程
    #print_Ndb(s)
    weight=readfile(s)   #对负数据库记录求期望
    return weight


