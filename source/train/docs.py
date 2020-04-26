from deepmd.Fitting import EnerFitting, WFCFitting, PolarFittingLocFrame, PolarFittingSeA, GlobalPolarFittingSeA, DipoleFittingSeA
from deepmd.DescrptLocFrame import DescrptLocFrame
from deepmd.DescrptSeA import DescrptSeA
from deepmd.DescrptSeR import DescrptSeR
from deepmd.DescrptSeAR import DescrptSeAR
from deepmd.Model import Model, TensorModel
from deepmd.Loss import EnerStdLoss, EnerDipoleLoss, TensorLoss
from deepmd.LearningRate import LearningRateExp
from deepmd.Trainer import NNPTrainer


class ParamArgs:
    def __init__(self, args, name, *subargs, depth=1, prefix=""):
        self.args = args
        self.name = name
        self.depth = depth
        self.subargs = subargs
        self.prefix = prefix

    @property
    def docs(self):
        d = [
            "#"*self.depth,
            " ",
            self.name,
            "\n",
            self.prefix,
            "\n",
            self.args.docs if self.args is not None else "",
            "\n",
        ]
        for sa in self.subargs:
            sa.depth = self.depth + 1
            d.append(sa.docs)
        return "".join(d)


def print_docs():
    params = ParamArgs(None, "Params",
        ParamArgs(None, "descriptor",
            ParamArgs(DescrptLocFrame.args, 'loc_frame'),
            ParamArgs(DescrptSeA.args, 'se_a'),
            ParamArgs(DescrptSeR.args, 'se_r'),
            ParamArgs(DescrptSeAR.args, 'se_ar'),
        ),
        ParamArgs(None, "fitting_net",
            ParamArgs(EnerFitting.args, "ener"),
            #ParamArgs(WFCFitting.args, "wfc"),
            #ParamArgs(None, "dipole",
            #    ParamArgs(DipoleFittingSeA.args, "se_a"),
            #),
            #ParamArgs(None, "polar",
            #    ParamArgs(PolarFittingLocFrame.args, "loc_frame"),
            #    ParamArgs(PolarFittingSeA.args, "se_a"),
            #),
            #ParamArgs(None, "global_polar",
            #    ParamArgs(GlobalPolarFittingSeA.args, "se_a"),
            #),
        ),
        ParamArgs(None, "model",
            ParamArgs(Model.args, "default"),
            ParamArgs(TensorModel.args, "wfc, dipole, polar, global_polar"),
        ),
        ParamArgs(None, "learning_rate",
            ParamArgs(LearningRateExp.args, "exp"),
        ),
        ParamArgs(None, "loss",
            ParamArgs(None, "ener",
                ParamArgs(EnerStdLoss.args, "std"),
                ParamArgs(EnerDipoleLoss.args, "ener_dipole"),
            ),
            #ParamArgs(TensorLoss.args, "wfc, dipole, polar, global_polar")
        ),
        ParamArgs(NNPTrainer.tr_args, "training")
    )

    with open("params.md", 'w') as f:
        f.write(params.docs)
