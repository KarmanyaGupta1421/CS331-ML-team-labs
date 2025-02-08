import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

aa = [1, 5, 10, 20, 80, 100, 1000]*7
values = [1,5,10,20,80,100,1000]
bb = [1, 5, 10, 20, 80, 100, 1000]*7
aa.sort()
print(len(aa))
# bb.sort()
# print(bb)

# idx = values.index(bb[1])
# print(idx)

# print(aa)

zz = [0.15026307644692047, 0.1530454967502321, 0.15638502011761063, 0.1580547818012999, 0.168628907458991, 0.16807335190343548, 0.18309192200557103, 0.15081708449396475, 0.15360259981429897, 0.15527081398947695, 0.15749767873723303, 0.16807180439492417, 0.16807180439492417, 0.18142061281337046, 0.15192974311358715, 0.1569374806561436, 0.15749767873723303, 0.1580578768183225, 0.16695914577530177, 0.16862581244196845, 0.1808635097493036, 0.1519281956050758, 0.158050139275766, 0.16139121015165583, 0.15861343237387807, 0.16584184463014545, 0.16973692355307954, 0.1808619622407923, 0.15359950479727638, 0.1608325595790777, 0.16472763850201175, 0.16584648715567935, 0.16862426493345714, 0.16862426493345714, 0.1752955741256577, 0.15304240173320952, 0.15971990095945526, 0.16472763850201178, 0.16917982048901264, 0.17029402661714638, 0.1736335499845249, 0.1786366450015475, 0.18309656453110493, 0.19477561126586196, 0.20479727638502015, 0.2253930671618694, 0.2626663571649644, 0.27101361807489943, 0.24429743113587127]
print(len(zz))
print("here")
aaa = []
bbb = []
ccc = []

for i in range(49):
    if (aa[i] != 1000 and bb[i] != 1000):
        aaa.append(aa[i])
        bbb.append(bb[i])
        ccc.append(zz[i])

fig = plt.figure()
ax = plt.axes(projection = "3d")

col = ['red', 'pink', 'blue', 'black', 'purple', 'orange', 'yellow']

# for i in range(0,49-7,7):
#     print(i)
#     ax.plot3D(aa[i:i+7], bb[i:i+7], zz[i:i+7], col[i//7])

for i in range(0, 36, 6):
    print(i)
    ax.plot3D(aaa[i:i+6], bbb[i:i+6], ccc[i:i+6], col[i//6])

# ax.plot3D(aa,bb,zz, 'red')
ax.set_xlabel('a value')
ax.set_ylabel('b value')
ax.set_zlabel('0-1 loss')

for a,b,c in zip(aa,bb,zz):
    print(a,b,c)

def rotate(angle):
    ax.view_init(azim = angle)
animat = anim.FuncAnimation(fig, rotate, frames=np.arange(0,350, 2), interval = 100)
t = anim.save('fixed_a.gif', dpi = 80, writer = 'imagepick')
plt.show()
