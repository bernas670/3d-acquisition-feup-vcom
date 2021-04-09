# vcom
Code development for the Computer Vision course @FEUP


wi = c11*x + c12*y + c13*z + c14
wj = c21*x + c22*y + c23*z + c24
w  = c31*x + c32*y + c33*z + c34

(c31*x + c32*y + c33*z + c34) * i = c11*x + c12*y + c13*z + c14
(c31*x + c32*y + c33*z + c34) * j = c21*x + c22*y + c23*z + c24

mpp

ultima linha * i
k1 = mpp[2] * i

(k1) c31 * i | c32 * i | c33 * i | c34  * i

ultima linha * j
k2 = mpp[2] * j

(k2) c31 * j | c32 * j | c33 * j | c34  * j

k1 - primeira linha
k3 = k1 - mpp[0]

(k3) c31 * i - c11 | c32 * i - c12 | c33 * i - c13 | c34  * i - c14

k2 - segunda linha
k4 = k2 - mpp[1]

(k4) c31 * j - c21 | c32 * j - c22 | c33 * j - c23 | c34  * j - c24

```


a = [k3[1:3], k4[1:3]]
b = [-k3[3],-k4[3]]
```
