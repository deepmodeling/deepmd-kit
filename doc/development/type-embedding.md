# Atom Type Embedding

## Overview

Here is an overview of the DeePMD-kit algorithm. Given a specific centric atom, we can obtain the matrix describing its local environment, named $\mathcal R$. It consists of the distance between the centric atom and its neighbors, as well as a direction vector. We can embed each distance into a vector of $M_1$ dimension by an `embedding net`, so the environment matrix $\mathcal R$ can be embedded into matrix $\mathcal G$. We can thus extract a descriptor vector (of $M_1 \times M_2$ dim) of the centric atom from the $\mathcal G$ by some matrix multiplication, and put the descriptor into `fitting net` to get the predicted energy $E$. The vanilla version of DeePMD-kit builds `embedding net` and `fitting net` relying on the atom type, resulting in $O(N)$ memory usage. After applying atom type embedding, in DeePMD-kit v2.0, we can share one `embedding net` and one `fitting net` in total, which reduces training complexity largely.

## Preliminary

In the following chart, you can find the meaning of symbols used to clarify the atom-type embedding algorithm.

<!-- GitHub Markdown cannot render math in a table... -->

$i$: Type of centric atom

$j$: Type of neighbor atom

$s_{ij}$: Distance between centric atom and neighbor atom

$\mathcal G_{ij}(\cdot)$: Origin embedding net, take $s_{ij}$ as input and output embedding vector of $M_1$ dim

$\mathcal G(\cdot)$: Shared embedding net

$\text{Multi}(\cdot)$: Matrix multiplication and flattening, output the descriptor vector of $M_1\times M_2$ dim

$F_i(\cdot)$: Origin fitting net, take the descriptor vector as input and output energy

$F(\cdot)$: Shared fitting net

$A(\cdot)$: Atom type embedding net, input is atom type, the output is type embedding vector of dim `nchanl`

So, we can formulate the training process as follows.
Vanilla DeePMD-kit algorithm:

$$E = F_i( \text{Multi}( \mathcal G_{ij}( s_{ij} ) ) )$$

DeePMD-kit applying atom type embedding:

$$E = F( [ \text{Multi}( \mathcal G( [s_{ij}, A(i), A(j)] ) ), A(j)] )$$

or

$$E = F( [ \text{Multi}( \mathcal G( [s_{ij}, A(j)] ) ), A(j)] )$$

The difference between the two variants above is whether using the information of centric atom when generating the descriptor. Users can choose by modifying the `type_one_side` hyper-parameter in the input JSON file.

## How to use

A detailed introduction can be found at [`se_e2_a_tebd`](../model/train-se-e2-a-tebd.md). Looking for a fast start-up, you can simply add a `type_embedding` section in the input JSON file as displayed in the following, and the algorithm will adopt the atom type embedding algorithm automatically.
An example of `type_embedding` is like

```json
    "type_embedding":{
       "neuron":    [2, 4, 8],
       "resnet_dt": false,
       "seed":      1
    }
```

## Code Modification

Atom-type embedding can be applied to varied `embedding net` and `fitting net`, as a result, we build a class `TypeEmbedNet` to support this free combination. In the following, we will go through the execution process of the code to explain our code modification.

### trainer (train/trainer.py)

In trainer.py, it will parse the parameter from the input JSON file. If a `type_embedding` section is detected, it will build a `TypeEmbedNet`, which will be later input in the `model`. `model` will be built in the function `_build_network`.

### model (model/ener.py)

When building the operation graph of the `model` in `model.build`. If a `TypeEmbedNet` is detected, it will build the operation graph of `type embed net`, `embedding net` and `fitting net` by order. The building process of `type embed net` can be found in `TypeEmbedNet.build`, which output the type embedding vector of each atom type (of [$\text{ntypes} \times \text{nchanl}$] dimensions). We then save the type embedding vector into `input_dict`, so that they can be fetched later in `embedding net` and `fitting net`.

### embedding net (descriptor/se\*.py)

In `embedding net`, we shall take local environment $\mathcal R$ as input and output matrix $\mathcal G$. Functions called in this process by the order is

```
build -> _pass_filter -> _filter -> _filter_lower
```

`_pass_filter`: It will first detect whether an atom type embedding exists, if so, it will apply atom type embedding algorithm and doesn't divide the input by type.

`_filter`: It will call `_filter_lower` function to obtain the result of matrix multiplication ($\mathcal G^T\cdot \mathcal R$), do further multiplication involved in $\text{Multi}(\cdot)$, and finally output the result of descriptor vector of $M_1 \times M_2$ dim.

`_filter_lower`: The main function handling input modification. If type embedding exists, it will call `_concat_type_embedding` function to concat the first column of input $\mathcal R$ (the column of $s_{ij}$) with the atom type embedding information. It will decide whether to use the atom type embedding vector of the centric atom according to the value of `type_one_side` (if set **True**, then we only use the vector of the neighbor atom). The modified input will be put into the `fitting net` to get $\mathcal G$ for further matrix multiplication stage.

### fitting net (fit/ener.py)

In `fitting net`, it takes the descriptor vector as input, whose dimension is [natoms, $M_1\times M_2$]. Because we need to involve information on the centric atom in this step, we need to generate a matrix named `atype_embed` (of dim [natoms, nchanl]), in which each row is the type embedding vector of the specific centric atom. The input is sorted by type of centric atom, we also know the number of a particular atom type (stored in `natoms[2+i]`), thus we get the type vector of the centric atom. In the build phase of the fitting net, it will check whether type embedding exists in `input_dict` and fetch them. After that, call `embed_atom_type` function to look up the embedding vector for the type vector of the centric atom to obtain `atype_embed`, and concat input with it ([input, atype_embed]). The modified input goes through `fitting` net` to get predicted energy.

:::{note}
You can't apply the compression method while using atom-type embedding.
:::
