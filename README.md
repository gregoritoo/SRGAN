# SRGAN

Implementation of https://arxiv.org/pdf/1609.04802.pdf

Architecture
-----------

![SRGAN](/img/SRGAN.png)


Objectives functions 
--------------------

Full minmax game : </br>
min<sub>*G*</sub>max<sub>0</sub>*V*(*D*, *G*) = *E*<sub>*x* ∼ *P*<sub>*d**a**t**a*</sub>∂(*x*)</sub>log (*D*(*x*)) + *E*<sub>2 ∼ *P*(*z*)</sub>\[log (1 − *D*(*G*(*z*)))\]

For discriminator :
  - min(log (*D*(*G*(z)))
  - max(log (*D*(x))
  
For generator :
  - max(log (*D*(*G*(z)))

