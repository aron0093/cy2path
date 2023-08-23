### Factorial latent dynamic models trained on Markovian simulations of biological processes using scRNAseq. data.

<table border="0">
<tr >
<td><img align="left" src="https://user-images.githubusercontent.com/25486108/208702939-0f2e9339-0d1f-467a-934c-56d5db388f22.gif" width="6400" height="300"></td>
 
<td>With a transition probability matrix $T$ over observed states $O$ and assuming Markovian dynamics, <br /><br />

<p align=center> $P(o \mid i) = P(o \mid o_{i-1})$ </p>

For iteration $i$,

<p align=center> $P(o \mid i) = P(o \mid i=0) \cdot T^i$ </p>

The animation overlays $P(i \mid o)$ on a 2D UMAP embedding of the data ([Cerletti et. al. 2020](https://doi.org/10.1101/2020.12.22.423929)) Since we are interested in modelling the dynamics in a smaller latent state space, we factorise the MSM simulation,

<p align=center> $P(o \mid i) = \sum\limits_{s \in S} P(o \mid s,i) P(s \mid i)$ </p>

Assuming Markovian dynamics in the latent space aswell,

<p align=center> $P(o \mid i) = \sum\limits_{s_{i} \in S} P(o \mid s_{i}) \sum\limits_{s_{i-1} \in S} P(s_{i} \mid s_{i-1})$ </p>

Multiple independent chains in a common latent space can be modelled using conditional latent TPMs ([Ghahramani & Jordan 1997](https://doi.org/10.1023/A:1007425814087)),

<p align=center> $P(o \mid i) = \sum\limits_{s_{i} \in S} P(o \mid s_{i}) \sum\limits_{l \in L} P(l) \sum\limits_{s_{i-1} \in S} P(s_{i} \mid s_{i-1}, l)$ </p>
</td>
</tr>
</table>

### Citation

Claassen, M., & Gupta, R. (2023). Factorial state-space modelling for kinetic clustering and lineage inference. https://doi.org/10.1101/2023.08.21.554135
