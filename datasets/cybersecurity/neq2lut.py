#  This file is part of PolyLUT.
#
#  PolyLUT is a derivative work based on LogicNets,
#  which is licensed under the Apache License 2.0.

#  Copyright (C) 2021 Xilinx, Inc
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import os
from argparse import ArgumentParser

import torch
import random
import numpy as np
from torch.utils.data import DataLoader

from polylut.nn import (
    generate_truth_tables,
    lut_inference,
    module_list_to_verilog_module,
)
from polylut.synthesis import synthesize_and_get_resource_counts
from polylut.util import proc_postsynth_file

from train import configs, model_config, dataset_config, test
from dataset import get_preqnt_dataset
from models import UnswNb15NeqModel, UnswNb15LutModel

other_options = {
    "seed": 20,
    "cuda": None,
    "device": 1,
    "log_dir": None,
    "checkpoint": None,
    "add_registers": False,
}

if __name__ == "__main__":
    parser = ArgumentParser(
        description="Synthesize convert a PyTorch trained model into verilog"
    )
    parser.add_argument(
        "--arch",
        type=str,
        choices=configs.keys(),
        default="nid-lite",
        help="Specific the neural network model to use (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        metavar="N",
        help="Batch size for evaluation (default: %(default)s)",
    )
    parser.add_argument(
        "--input-bitwidth",
        type=int,
        default=None,
        help="Bitwidth to use at the input (default: %(default)s)",
    )
    parser.add_argument(
        "--hidden-bitwidth",
        type=int,
        default=None,
        help="Bitwidth to use for activations in hidden layers (default: %(default)s)",
    )
    parser.add_argument(
        "--output-bitwidth",
        type=int,
        default=None,
        help="Bitwidth to use at the output (default: %(default)s)",
    )
    parser.add_argument(
        "--input-fanin",
        type=int,
        default=None,
        help="Fanin to use at the input (default: %(default)s)",
    )
    parser.add_argument(
        "--degree",
        type=int,
        default=None,
        help="Degree to use for polynomials (default: %(default)s)",
    )
    parser.add_argument(
        "--hidden-fanin",
        type=int,
        default=None,
        help="Fanin to use for the hidden layers (default: %(default)s)",
    )
    parser.add_argument(
        "--output-fanin",
        type=int,
        default=None,
        help="Fanin to use at the output (default: %(default)s)",
    )
    parser.add_argument(
        "--hidden-layers",
        nargs="+",
        type=int,
        default=None,
        help="A list of hidden layer neuron sizes (default: %(default)s)",
    )
    parser.add_argument(
        "--clock-period",
        type=float,
        default=1.0,
        help="Target clock frequency to use during Vivado synthesis (default: %(default)s)",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Dataset to use for evaluation (default: %(default)s)",
    )
    parser.add_argument(
        "--dataset-file",
        type=str,
        default="data/unsw_nb15_binarized.npz",
        help="The file to use as the dataset input (default: %(default)s)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="0",
        help="A location to store the log output of the training run and the output model (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Seed to use for RNG (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        type=int,
        required=True,
        help="Device_id for GPU (default: %(default)s)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="The checkpoint file which contains the model weights",
    )
    parser.add_argument(
        "--add-registers",
        action="store_true",
        default=False,
        help="Add registers between each layer in generated verilog (default: %(default)s)",
    )

    args = parser.parse_args()
    defaults = configs[args.arch]
    options = vars(args)
    del options["arch"]
    config = {}
    for k in options.keys():
        config[k] = (
            options[k] if options[k] is not None else defaults[k]
        )  # Override defaults, if specified.

    if not os.path.exists(config["log_dir"]):
        os.makedirs(config["log_dir"])

    # Split up configuration options to be more understandable
    model_cfg = {}
    for k in model_config.keys():
        model_cfg[k] = config[k]
    dataset_cfg = {}
    for k in dataset_config.keys():
        dataset_cfg[k] = config[k]
    options_cfg = {}
    for k in other_options.keys():
        if k == "cuda":
            continue
        options_cfg[k] = config[k]

    # Set random seeds
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    os.environ["PYTHONHASHSEED"] = str(config["seed"])

    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(options_cfg["device"])

    # Fetch the test set
    dataset = {}
    dataset[args.dataset_split] = get_preqnt_dataset(
        dataset_cfg["dataset_file"], split=args.dataset_split
    )
    test_loader = DataLoader(
        dataset[args.dataset_split], batch_size=config["batch_size"], shuffle=False
    )

    # Instantiate the PyTorch model
    x, y = dataset[args.dataset_split][0]
    model_cfg["input_length"] = len(x)
    model_cfg["output_length"] = 1
    model = UnswNb15NeqModel(model_cfg)
    model.cuda()

    # Load the model weights
    checkpoint = torch.load(options_cfg['checkpoint'], map_location="cuda:{}".format(options_cfg["device"]))
    model.load_state_dict(checkpoint['model_dict'])

   # Test the PyTorch model
    print("Running inference on baseline model...")
    baseline_accuracy = test(model, test_loader, cuda=True)
    print("Baseline accuracy: %f" % (baseline_accuracy))

    # Generate the truth tables in the LUT module
    print("Converting to NEQs to LUTs...")
    generate_truth_tables(model, verbose=True)

    # Test the LUT-based model
    print("Running inference on LUT-based model...")
    lut_inference(model)
    lut_accuracy = test(model, test_loader, cuda=True)
    print("LUT-Based Model accuracy: %f" % (lut_accuracy))
    modelSave = {"model_dict": model.state_dict(), "test_accuracy": lut_accuracy}

    torch.save(modelSave, options_cfg["log_dir"] + "/lut_based_model.pth")
    print("Generating verilog in %s..." % (options_cfg["log_dir"]))
    module_list_to_verilog_module(
        model.module_list,
        "polylut",
        options_cfg["log_dir"],
        add_registers=options_cfg["add_registers"],
    )
    print("Top level entity stored at: %s/polylut.v ..." % (options_cfg["log_dir"]))