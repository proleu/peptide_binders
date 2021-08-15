#!/mnt/home/pleung/.conda/envs/crispy/bin/python
__author__ = "Philip Leung, Minkyung Baek, Ivan Anishchenko, Sergey Ovchinnikov"
__copyright__ = None
__credits__ = ["Philip Leung", "Minkyung Baek", "Ivan Anishchenko", "Sergey Ovchinnikov", "Rosettacommons"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "Philip Leung"
__email__ = "pleung@cs.washington.edu"
__status__ = "Prototype"
import argparse, binascii, bz2, collections, json, os, pyrosetta, sys
from pyrosetta.distributed import cluster
import pyrosetta.distributed.io as io
from pyrosetta.distributed.packed_pose.core import PackedPose
from typing import *


parser = argparse.ArgumentParser(description="Use to run AF2.")
# required arguments
parser.add_argument("-s", type=str, default="", required=True)


def run_af2(
    prefix="",  # prefix for saving pdbs, can include path components
    query="",  # relative or abspath to pdb, pdb.gz, or pdb.bz2
    num_recycle=3,  # set this to 10 if plddts are low - might help models converge
    random_seed=0,  # try changing seed if you need to sample more
    num_models=5,  # it will run [4, 3, 5, 2, 1][:num_models] these models, 4 is used for compiling jax params
    index_gap=200,  # decrease under 32 if you have a prior that chains need to interact
    save_pdbs=True,  # if false will save pdbstring in output dict instead
) -> Dict:
    import bz2
    import os

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    from string import ascii_uppercase
    import sys

    sys.path.insert(0, "/projects/ml/alphafold/alphafold_git/")
    from typing import Dict
    import jax
    from jax.lib import xla_bridge
    import matplotlib.pyplot as plt

    import numpy as np
    from alphafold.common import protein
    from alphafold.data import pipeline
    from alphafold.data import templates
    from alphafold.model import data
    from alphafold.model import config
    from alphafold.model import model
    from alphafold.relax import relax
    from alphafold.relax import utils
    import pyrosetta
    import pyrosetta.distributed.io as io
    from pyrosetta.distributed.tasks.rosetta_scripts import (
        SingleoutputRosettaScriptsTask,
    )
    from pyrosetta.rosetta.core.pose import Pose

    def set_bfactor(pose: Pose, lddt_array: list) -> None:
        for resid, residue in enumerate(pose.residues, start=1):
            for i, atom in enumerate(residue.atoms(), start=1):
                pose.pdb_info().bfactor(resid, i, lddt_array[resid - 1])
        return

    def mk_mock_template(query_sequence: str) -> Dict:
        """
        Make a mock template dict from a query sequence.
        Since alphafold's model requires a template input,
        we create a blank example w/ zero input, confidence -1
        @minkbaek @aivan
        """
        ln = len(query_sequence)
        output_templates_sequence = "-" * ln
        output_confidence_scores = np.full(ln, -1)
        templates_all_atom_positions = np.zeros(
            (ln, templates.residue_constants.atom_type_num, 3)
        )
        templates_all_atom_masks = np.zeros(
            (ln, templates.residue_constants.atom_type_num)
        )
        templates_aatype = templates.residue_constants.sequence_to_onehot(
            output_templates_sequence, templates.residue_constants.HHBLITS_AA_TO_ID
        )
        template_features = {
            "template_all_atom_positions": templates_all_atom_positions[None],
            "template_all_atom_masks": templates_all_atom_masks[None],
            "template_sequence": [f"none".encode()],
            "template_aatype": np.array(templates_aatype)[None],
            "template_confidence_scores": output_confidence_scores[None],
            "template_domain_names": [f"none".encode()],
            "template_release_date": [f"none".encode()],
        }
        return template_features

    def get_rmsd(design: Pose, prediction: Pose) -> float:
        """Calculate Ca-RMSD of prediction to design"""
        rmsd_calc = pyrosetta.rosetta.core.simple_metrics.metrics.RMSDMetric()
        rmsd_calc.set_rmsd_type(pyrosetta.rosetta.core.scoring.rmsd_atoms(3))
        rmsd_calc.set_run_superimpose(True)
        rmsd_calc.set_comparison_pose(design)
        rmsd = float(rmsd_calc.calculate(prediction))
        return rmsd

    def DAN(pdb: str) -> np.array:
        import os, subprocess

        def cmd(command, wait=True):
            """@nrbennet @bcov"""
            the_command = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
            if not wait:
                return
            the_stuff = the_command.communicate()
            return str(the_stuff[0]) + str(the_stuff[1])

        pythonpath = "/software/conda/envs/tensorflow/bin/python"
        script = "/net/software/DeepAccNet/DeepAccNet.py"
        npz = pdb.replace(".pdb", ".npz")
        to_send = f"""{pythonpath} {script} -r -v --pdb {pdb} {npz} """
        print(cmd(to_send))
        x = np.load(npz)
        os.remove(npz)
        lddt = x["lddt"]
        return lddt

    def predict_structure(
        prefix="",
        feature_dict={},
        Ls=[],
        model_params={},
        use_model={},
        random_seed=0,
        index_gap=200,
        save_pdbs=True,
    ) -> Dict:
        """Predicts structure using AlphaFold for the given pdb/pdb.bz2."""
        # Minkyung"s code adds big enough number to residue index to indicate chain breaks
        idx_res = feature_dict["residue_index"]
        L_prev = 0
        # Ls: number of residues in each chain
        for L_i in Ls[:-1]:
            idx_res[L_prev + L_i :] += index_gap
            L_prev += L_i
        feature_dict["residue_index"] = idx_res
        # Run the models.
        plddts, paes, ptms, rmsds = [], [], [], []
        poses = []
        for model_name, params in model_params.items():
            if model_name in use_model:
                model_runner = (
                    model_runner_4  # global variable, only model 4 is compiled
                )
                model_runner.params = params
                processed_feature_dict = model_runner.process_features(
                    feature_dict, random_seed=random_seed
                )
                prediction_result = model_runner.predict(processed_feature_dict)
                unrelaxed_protein = protein.from_prediction(
                    processed_feature_dict, prediction_result
                )
                plddts.append(prediction_result["plddt"])
                paes.append(prediction_result["predicted_aligned_error"])
                ptms.append(prediction_result["ptm"])
                # add termini after each chain
                unsafe_pose = io.to_pose(
                    io.pose_from_pdbstring(protein.to_pdb(unrelaxed_protein))
                )
                cleaned_pose = Pose()
                total = 0
                chunks = []
                mylist = list(unsafe_pose.residues)
                for j in range(len(Ls)):
                    chunk_mylist = mylist[total : total + Ls[j]]
                    chunks.append(chunk_mylist)
                    total += Ls[j]
                    temp_pose = Pose()
                    for k in chunk_mylist:
                        temp_pose.append_residue_by_bond(k)
                    pyrosetta.rosetta.core.pose.append_pose_to_pose(
                        cleaned_pose, temp_pose, True
                    )
                sc = pyrosetta.rosetta.protocols.simple_moves.SwitchChainOrderMover()
                sc.chain_order("".join([str(i) for i in range(1, len(Ls) + 1)]))
                sc.apply(cleaned_pose)
                rmsds.append(get_rmsd(pose, cleaned_pose))
                # relax sidechains to prevent distracting clashes in output
                xml = """
                <ROSETTASCRIPTS>
                    <SCOREFXNS>
                        <ScoreFunction name="sfxn" weights="beta_nov16" />
                    </SCOREFXNS>
                    <RESIDUE_SELECTORS>
                    </RESIDUE_SELECTORS>
                    <TASKOPERATIONS>
                    </TASKOPERATIONS>
                    <TASKOPERATIONS>
                        <IncludeCurrent name="current" />
                    </TASKOPERATIONS>
                    <MOVERS>
                        <FastRelax name="relax" scorefxn="sfxn" repeats="1" bondangle="false" bondlength="false" task_operations="current" >
                            <MoveMap name="MM" bb="false" chi="true" jump="false" />
                        </FastRelax>
                    </MOVERS>
                    <FILTERS>
                    </FILTERS>
                    <APPLY_TO_POSE>
                    </APPLY_TO_POSE>
                    <PROTOCOLS>
                        <Add mover="relax" />
                    </PROTOCOLS>
                    <OUTPUT />
                </ROSETTASCRIPTS>
                """
                relaxer = SingleoutputRosettaScriptsTask(xml)
                relaxed_ppose = relaxer(cleaned_pose.clone())
                poses.append(io.to_pose(relaxed_ppose))

        model_idx = [4, 3, 5, 1, 2]
        model_idx = model_idx[:num_models]
        out = {}
        # save output pdbs and metadata
        for n, r in enumerate(model_idx):
            os.makedirs(
                os.path.join(os.getcwd(), "/".join(prefix.split("/")[:-1])),
                exist_ok=True,
            )
            relaxed_pdb_path = f"{prefix}_relaxed_model_{r}.pdb"
            set_bfactor(poses[n], list(plddts[n]))
            poses[n].dump_pdb(relaxed_pdb_path)
            average_plddts = float(plddts[n].mean())

            out[f"model_{r}"] = {
                "average_plddts": average_plddts,
                "plddt": plddts[n].tolist(),
                "pae": paes[n].tolist(),
                "ptm": ptms[n].tolist(),
                "rmsd_to_input": rmsds[n],
                "pdb_path": os.path.abspath(relaxed_pdb_path),
            }
            print(f"model_{r}: average plddt {average_plddts}")
        return out

    # begin main method
    pyrosetta.init("-run:constant_seed 1 -mute all -corrections::beta_nov16 true")
    # read in pdbs, do bz2 check, if query does not contain .pdb throw exception
    if ".pdb" in query and ".bz2" not in query:
        pose = pyrosetta.io.pose_from_file(query)
    elif ".pdb.bz2" in query:
        with open(query, "rb") as f:
            ppose = io.pose_from_pdbstring(bz2.decompress(f.read()).decode())
        pose = io.to_pose(ppose)
    else:
        raise RuntimeError("query must be a pdb, pdb.gz, or pdb.bz2")
    n_chains = pose.num_chains()
    seqs = [chain.sequence() for chain in pose.split_by_chain()]
    full_sequence = "".join(seqs)
    # prepare models
    use_model = {}
    if "model_params" not in dir():
        model_params = {}
    for model_name in ["model_4", "model_3", "model_5", "model_1", "model_2"][
        :num_models
    ]:
        use_model[model_name] = True
        if model_name not in model_params:
            model_params[model_name] = data.get_model_haiku_params(
                model_name=model_name + "_ptm",
                data_dir="/projects/ml/alphafold/alphafold_git/",
            )
            if (
                model_name == "model_4"
            ):  # compile only model 4 and later load weights for other models
                model_config = config.model_config(model_name + "_ptm")
                model_config.data.common.max_extra_msa = 1
                model_config.data.eval.max_msa_clusters = n_chains
                model_config.data.eval.num_ensemble = 1
                model_config.data.common.num_recycle = num_recycle
                model_runner_4 = model.RunModel(model_config, model_params[model_name])
    # prepare input data
    template_features = mk_mock_template(full_sequence)  # make mock template
    deletion_matrix = [[0] * len(full_sequence)]  # make mock deletion matrix
    msas = []
    deletion_matrices = []
    for i in range(n_chains):
        # make a sequence of length full_sequence where everything but the i-th chain is "-"
        msa = [
            "".join(["-" * len(seq) if i != j else seq for j, seq in enumerate(seqs)])
        ]
        msas.append(msa)
        deletion_matrices.append(deletion_matrix)
    feature_dict = {
        **pipeline.make_sequence_features(
            sequence=full_sequence,
            description="none",
            num_res=len(full_sequence),
        ),
        **pipeline.make_msa_features(msas=msas, deletion_matrices=deletion_matrices),
        **template_features,
    }
    # predict structure
    if prefix == "":
        prefix = query
    else:
        pass
    out = predict_structure(
        prefix=prefix,
        feature_dict=feature_dict,
        Ls=[len(l) for l in seqs],
        model_params=model_params,
        use_model=use_model,
        random_seed=random_seed,
        index_gap=index_gap,
        save_pdbs=save_pdbs,
    )
    # deallocate backend memory to make room for DAN
    device = xla_bridge.get_backend().platform
    backend = xla_bridge.get_backend(device)
    for buffer in backend.live_buffers():
        buffer.delete()
    # run DAN
    for model, result in out.items():
        pdb_path = result["pdb_path"]
        DAN_plddt = DAN(pdb_path)
        result["average_DAN_plddts"] = float(DAN_plddt.mean())
        result["DAN_plddt"] = DAN_plddt.tolist()
        # if not save, write pdbstrings to output dict
        if not save_pdbs:
            result["pdb_string"] = io.to_pdbstring(io.pose_from_file(pdb_path))
            os.remove(pdb_path)
    return out

def main():
    if len(sys.argv) == 1:
        parser.print_help()
    else:
        pass
    args = parser.parse_args(sys.argv[1:])
    print("Inference will proceed with the following options:")
    print(args)

    run_kwargs = vars(args)
    run_kwargs["-s"] = run_kwargs.pop("s")
    with open(run_kwargs["-s"], "rb") as f:
        temp_ppose = io.pose_from_pdbstring(bz2.decompress(f.read()).decode())
    pose = io.to_pose(temp_ppose)
    pose = pose.split_by_chain(1)
    handle = str(binascii.b2a_hex(os.urandom(24)).decode("utf-8"))
    pose.dump_pdb(f"{handle}.pdb")
    metadata = run_af2(save_pdbs=False,query=f"{handle}.pdb")
    if metadata is not None:
        os.remove(f"{handle}.pdb")
        print(metadata) # TODO
        json_string = json.dumps(metadata)
        output_file = run_kwargs["-s"].replace(".pdb.bz2", ".json")
        with open(output_file, "w+") as f:
            print(json_string, file=f)

if __name__ == "__main__":
    main()
