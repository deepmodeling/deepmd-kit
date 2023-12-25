import os
from os import path as osp

import paddle
import paddle_deepmd_lib

unitest_dir = os.getenv("UNITTEST_DIR", None)

if unitest_dir is None:
    raise ValueError(
        "Please download unitest data and set env with 4 scipts below:\n"
        "1. wget -nc https://paddle-org.bj.bcebos.com/paddlescience/deepmd/deepmd_custom_op_test_data.tar\n"
        "2. tar -xf deepmd_custom_op_test_data.tar\n"
        "3. export UNITTEST_DIR=$PWD/deepmd_custom_op_test_data\n"
        "4. python ./custom_op_test.py\n"
    )


def test_neighbor_stat(place="cpu"):
    print("=" * 10, f"test_neighbor_stat [place={place}]", "=" * 10)
    import numpy as np

    coord = np.ascontiguousarray(
        np.load(osp.join(unitest_dir, "neighbor_stat/coord.npy"))
    )
    type = np.ascontiguousarray(
        np.load(osp.join(unitest_dir, "neighbor_stat/type.npy"))
    )
    natoms = np.ascontiguousarray(
        np.load(osp.join(unitest_dir, "neighbor_stat/natoms_vec.npy"))
    )
    box = np.ascontiguousarray(np.load(osp.join(unitest_dir, "neighbor_stat/box.npy")))
    default_mesh = np.ascontiguousarray(
        np.load(osp.join(unitest_dir, "neighbor_stat/default_mesh.npy"))
    )

    rcut = 6.0

    coord = paddle.to_tensor(coord, "float64", place=place)
    type = paddle.to_tensor(type, "int32", place=place)
    natoms = paddle.to_tensor(natoms, "int32", place=place)
    box = paddle.to_tensor(box, "float64", place=place)
    default_mesh = paddle.to_tensor(default_mesh, "int32", place=place)

    mn, dt = paddle_deepmd_lib.neighbor_stat(
        coord,
        type,
        natoms,
        box,
        default_mesh,
        rcut=rcut,
    )

    mn_load = np.load(osp.join(unitest_dir, "neighbor_stat/mn.npy"))
    dt_load = np.load(osp.join(unitest_dir, "neighbor_stat/dt.npy"))

    # print(mn.shape, mn.min().item()); print(mn.max().item()); print(mn.mean().item()); print(mn.var().item())
    # print(mn_load.shape); print(mn_load.min().item()); print(mn_load.max().item()); print(mn_load.mean().item()); print(mn_load.var().item())
    # print(dt.shape, dt.min().item(), dt.max().item(), dt.mean().item(), dt.var().item())
    # print(dt_load.shape, dt_load.min().item(), dt_load.max().item(), dt_load.mean().item(), dt_load.var().item())

    print(np.allclose(mn.numpy(), mn_load))
    print(np.allclose(dt.numpy(), dt_load))


def test_prod_env_mat_a(place="cpu"):
    print("=" * 10, f"test_prod_env_mat_a [place={place}]", "=" * 10)
    import numpy as np

    # "coord", "atype", "natoms", "box", "mesh", "t_avg", "t_std"
    coord = np.ascontiguousarray(
        np.load(osp.join(unitest_dir, "prod_env_mat_a/descrpt.coord.npy"))
    )
    atype = np.ascontiguousarray(
        np.load(osp.join(unitest_dir, "prod_env_mat_a/descrpt.atype.npy"))
    )
    natoms = np.ascontiguousarray(
        np.load(osp.join(unitest_dir, "prod_env_mat_a/descrpt.natoms.npy"))
    )
    box = np.ascontiguousarray(
        np.load(osp.join(unitest_dir, "prod_env_mat_a/descrpt.box.npy"))
    )
    mesh = np.ascontiguousarray(
        np.load(osp.join(unitest_dir, "prod_env_mat_a/descrpt.mesh.npy"))
    )
    t_avg = np.ascontiguousarray(
        np.load(osp.join(unitest_dir, "prod_env_mat_a/descrpt.t_avg.npy"))
    )
    t_std = np.ascontiguousarray(
        np.load(osp.join(unitest_dir, "prod_env_mat_a/descrpt.t_std.npy"))
    )
    t_std = np.ascontiguousarray(
        np.load(osp.join(unitest_dir, "prod_env_mat_a/descrpt.t_std.npy"))
    )

    coord = paddle.to_tensor(coord, place=place)
    atype = paddle.to_tensor(atype, place=place)
    natoms = paddle.to_tensor(natoms, place="cpu")
    box = paddle.to_tensor(box, place=place)
    mesh = paddle.to_tensor(mesh, place=place)
    t_avg = paddle.to_tensor(t_avg, place=place)
    t_std = paddle.to_tensor(t_std, place=place)

    rcut_a = -1
    rcut_r = 6.0
    rcut_r_smth = 0.5
    sel_a = [46, 92]
    sel_r = [0, 0]

    # print(coord.shape, coord.dtype, coord.place)
    # print(atype.shape, atype.dtype, atype.place)
    # print(box.shape, box.dtype, box.place)
    # print(mesh.shape, mesh.dtype, mesh.place)
    # print(t_avg.shape, t_avg.dtype, t_avg.place)
    # print(t_std.shape, t_std.dtype, t_std.place)
    # print(natoms.shape, natoms.dtype, natoms.place)
    # print(rcut_a)
    # print(rcut_r)
    # print(rcut_r_smth)
    # print(sel_a)
    # print(sel_r)

    descrpt, descrpt_deriv, rij, nlist = paddle_deepmd_lib.prod_env_mat_a(
        coord,
        atype,
        box,
        mesh,
        t_avg,
        t_std,
        natoms,
        rcut_a,
        rcut_r,
        rcut_r_smth,
        sel_a,
        sel_r,
    )
    descrpt_load = np.load(osp.join(unitest_dir, "prod_env_mat_a/descrpt.descrpt.npy"))
    descrpt_deriv_load = np.load(
        osp.join(unitest_dir, "prod_env_mat_a/descrpt.descrpt_deriv.npy")
    )
    rij_load = np.load(osp.join(unitest_dir, "prod_env_mat_a/descrpt.rij.npy"))
    nlist_load = np.load(osp.join(unitest_dir, "prod_env_mat_a/descrpt.nlist.npy"))
    # print(descrpt.shape)
    # print(descrpt_deriv.shape)
    # print(rij.shape)
    # print(nlist.shape)
    # print(descrpt_load.shape) # (1, 576)
    # print(descrpt_deriv_load.shape) # (1, 192)
    # print(rij_load.shape) # (4,)
    # print(nlist_load.shape) # (1, 9)

    print(np.allclose(descrpt.numpy(), descrpt_load))
    print(np.allclose(descrpt_deriv.numpy(), descrpt_deriv_load))
    print(np.allclose(rij.numpy(), rij_load))
    print(np.allclose(nlist.numpy(), nlist_load))


def test_prod_force_se_a(place="cpu"):
    print("=" * 10, f"test_prod_force_se_a [place={place}]", "=" * 10)
    import numpy as np

    # "coord", "atype", "natoms", "box", "mesh", "t_avg", "t_std"
    net_deriv_reshape = np.ascontiguousarray(
        np.load(osp.join(unitest_dir, "prod_force_se_a/descrpt.net_deriv_reshape.npy"))
    )
    descrpt_deriv = np.ascontiguousarray(
        np.load(osp.join(unitest_dir, "prod_force_se_a/descrpt.descrpt_deriv.npy"))
    )
    nlist = np.ascontiguousarray(
        np.load(osp.join(unitest_dir, "prod_force_se_a/descrpt.nlist.npy"))
    )
    natoms = np.ascontiguousarray(
        np.load(osp.join(unitest_dir, "prod_force_se_a/descrpt.natoms.npy"))
    )

    nnei_a = 138
    nnei_r = 0

    net_deriv_reshape = paddle.to_tensor(
        net_deriv_reshape, stop_gradient=False, place=place
    )
    descrpt_deriv = paddle.to_tensor(descrpt_deriv, place=place)
    nlist = paddle.to_tensor(nlist, place=place)
    natoms = paddle.to_tensor(natoms, place="cpu")  # [192, 192, 64 , 128]
    force = paddle_deepmd_lib.prod_force_se_a(
        net_deriv_reshape,
        descrpt_deriv,
        nlist,
        natoms,
        n_a_sel=nnei_a,
        n_r_sel=nnei_r,
    )
    force.sum().backward()
    # print(f"net_deriv_reshape.grad.shape = {net_deriv_reshape.grad.shape}")

    force_load = np.load(osp.join(unitest_dir, "prod_force_se_a/descrpt.force.npy"))
    # print(force.shape) # (1, 9)
    # print(force_load.shape) # (1, 9)

    print(np.allclose(force.numpy(), force_load))


def test_prod_virial_se_a(place="cpu"):
    print("=" * 10, f"test_prod_virial_se_a [place={place}]", "=" * 10)
    import numpy as np

    # "coord", "atype", "natoms", "box", "mesh", "t_avg", "t_std"
    net_deriv_reshape = np.ascontiguousarray(
        np.load(osp.join(unitest_dir, "prod_virial_se_a/descrpt.net_deriv_reshape.npy"))
    )
    descrpt_deriv = np.ascontiguousarray(
        np.load(osp.join(unitest_dir, "prod_virial_se_a/descrpt.descrpt_deriv.npy"))
    )
    rij = np.ascontiguousarray(
        np.load(osp.join(unitest_dir, "prod_virial_se_a/descrpt.rij.npy"))
    )
    nlist = np.ascontiguousarray(
        np.load(osp.join(unitest_dir, "prod_virial_se_a/descrpt.nlist.npy"))
    )
    natoms = np.ascontiguousarray(
        np.load(osp.join(unitest_dir, "prod_virial_se_a/descrpt.natoms.npy"))
    )

    nnei_a = 138
    nnei_r = 0

    net_deriv_reshape = paddle.to_tensor(
        net_deriv_reshape, stop_gradient=False, place=place
    )
    descrpt_deriv = paddle.to_tensor(descrpt_deriv, place=place)
    rij = paddle.to_tensor(rij, place=place)
    nlist = paddle.to_tensor(nlist, place=place)
    natoms = paddle.to_tensor(natoms, place="cpu")
    virial, atom_virial = paddle_deepmd_lib.prod_virial_se_a(
        net_deriv_reshape,
        descrpt_deriv,
        rij,
        nlist,
        natoms,
        n_a_sel=nnei_a,
        n_r_sel=nnei_r,
    )
    virial.sum().backward()
    # print(f"net_deriv_reshape.grad.shape = {net_deriv_reshape.grad.shape}")

    virial_load = np.load(osp.join(unitest_dir, "prod_virial_se_a/descrpt.virial.npy"))
    atom_virial_load = np.load(
        osp.join(unitest_dir, "prod_virial_se_a/descrpt.atom_virial.npy")
    )
    # print(virial.shape) # (1, 9)
    # print(virial_load.shape) # (1, 9)
    # print(atom_virial.shape) # (1, 9)
    # print(atom_virial_load.shape) # (1, 9)

    print(np.allclose(virial.numpy(), virial_load))
    print(np.allclose(atom_virial.numpy(), atom_virial_load))


if __name__ == "__main__":
    test_neighbor_stat()

    test_prod_env_mat_a("gpu")
    test_prod_force_se_a("gpu")
    test_prod_virial_se_a("gpu")

    test_prod_env_mat_a("cpu")
    test_prod_force_se_a("cpu")
    test_prod_virial_se_a("cpu")
