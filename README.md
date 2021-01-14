# SRGAN

Implementation of https://arxiv.org/pdf/1609.04802.pdf

Architecture
-----------

![SRGAN](/img/SRGAN.png)


Objectives functions 
--------------------

Full minmax game : </br>
min<sub>*G*</sub>max<sub>D</sub>*V*(*D*, *G*) = *E*<sub>*x* ∼ *P*<sub>*data</sub>(*x*)</sub>log (*D*(*x*)) + *E*<sub>z ∼ *P*(*z*)</sub>\[log (1 − *D*(*G*(*z*)))\]

For discriminator :
  - min(log (*D*(*G*(z)))
  - max(log (*D*(x))
  
For generator :
  - max(log (*D*(*G*(z)))

