

from typing import Union, Any, Dict, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pathlib import Path


import logging
logger = logging.getLogger(__name__)


def load_checkpoint(model_path: Path, jit: bool = True) -> Dict:
    checkpoint = {}
    model = torch.jit.load(str(model_path), map_location='cpu')
    checkpoint['model'] = model
    return checkpoint



def save_checkpoint(model: Union[nn.Module, torch.jit.ScriptModule],
                    output_model_path: Path,
                    epoch: int,
                    save_torch: bool = False,
                    use_breakpoint: bool = False,
                    ) -> None:


    if not output_model_path.parent.exists():
        output_model_path.parent.mkdir(parents=True, exist_ok=True)


    model_script = None
    try:
        model_script = torch.jit.script(model)
    except Exception as e:
        print(f"======================================= {e}")
        logger.error("Error during jit scripting, so save torch state_dict.")
        if use_breakpoint:
            breakpoint()


    save_path = None

    if model_script is not None:
        try:
            save_path = output_model_path.parent / f'{output_model_path.name}.jit'
            torch.jit.save(model_script, str(save_path))
            logger.warning(f"Saved torchscript model at {save_path}")
        except:
            breakpoint()

    if save_torch:
        save_torch_path = output_model_path
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
        }, save_torch_path)
        logger.warning(f"Saved torch model at {save_torch_path}")

    return save_path