### Factorial latent dynamic models trained on Markovian simulations of biological processes using scRNAseq. data.

<table border="0">
<tr >
<td><img align="left" src="https://user-images.githubusercontent.com/25486108/208702939-0f2e9339-0d1f-467a-934c-56d5db388f22.gif" width="6400" height="300"></td>
 
 <td>We begin with a transition probability matrix of cell states. Assuming Markovian dynamics,

<p align=center> $P(cell|t) = P(cell|cell_{t-1})$ </p>

For iteration $t$,

<p align=center> $P(cell|t, init) = P(cell|init) \cdot TPM^t$ </p>

The animation overlays $P(t|cell,init)$ on a 2D UMAP embedding of the data ([Cerletti et. al. 2020](https://www.biorxiv.org/content/10.1101/2020.12.22.423929v1)) Since we are interested in modelling lineages we factorise the MSM simulation like so,

<p align=center> $P(cell|t) = \sum_l \sum_s P(cell|s,l,t) P(l|s,t) P(s|t)$ </p>

Assuming Markovian dynamics in a common latent space,

<p align=center> $P(cell|t) = \sum_l \sum_s P(cell|s_{t}) P(l|s_{t}) P(s_{t}|s_{t-1})$ </p>
</td>
</tr>
</table>
