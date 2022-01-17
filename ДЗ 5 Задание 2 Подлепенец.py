import numpy as np

def act(x):
    return 0 if x < 1 else 1

def go(act, lina, zdr, igr, las, ob, dr):
    x = np.array([act, lina, zdr, igr, las, ob, dr])
    w1 = [1, 0, 0, 0, 1]
    w2 = [1, 0, 0, 0, 1]
    w3 = [0, 0, 1, 0, 1]
    w4 = [1, 1, 1, 0, 0]
    w5 = [0, 0, 1, 1, 1]
    w6 = [0, 1, 0, 0, 0]
    w7 = [0, 0, 1, 0, 1]
    weight1 = np.array([w1, w2, w3, w4, w5, w6, w7])
    weight2 = np.array([1, 2, 3, 4, 5, 6, 7])

    sum_hidden = np.dot(weight1, x)
    print("Значения сумм на нейронах скрытого слоя: "+str(sum_hidden))

    out_hidden = np.array([act(x) for x in sum_hidden])
    print("Значения на выходах нейронов скрытого слоя: "+str(out_hidden))

    sum_end = np.dot(weight2, out_hidden)
    y = act(sum_end)
    print("Выходное значение НС: "+str(y))

    return y

act = 1
lina = 1
zdr = 0
igr = 1
las = 0
ob = 0
dr = 0

res = go(act, lina, zdr, igr, las, ob, dr)
if res == 1:
    print("Это сиамская кошка")
if res == 2:
    print("Это русская голубая кошка")
if res == 3:
    print("Это бирманская кошка")
if res == 4:
    print("Это шартрез кошка")
if res == 5:
    print("Это сфинкс кошка")

