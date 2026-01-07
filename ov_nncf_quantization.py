import openvino as ov
import nncf
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data_count=100, speech_length=100):
        self.dataset = []
        for i in range(data_count):
            #! funasr bb model
            # pixel_values = torch.randn(1, 4988, 3, 14, 14, dtype=torch.float32)
            # # cu_seqlens = torch.tensor([2], dtype=torch.int32)
            # cu_seqlens = torch.randint(0, 100, (2,), dtype=torch.int32)
            # image_grid_thw = torch.tensor([[1, 28, 28]], dtype=torch.int32)

            pixel_values = torch.rand((1, 4988, 3, 14, 14), dtype=torch.float32)
            image_grid_thw = torch.tensor([[1, 58, 86]], dtype=torch.int32)
            cu_seqlens = torch.tensor([0, 4988], dtype=torch.int32)
            self.dataset.append([pixel_values, cu_seqlens, image_grid_thw])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,idx):
        image = self.dataset[idx]
        return image

def transform_fn(data_item):
    inputs = {"pixel_values":data_item[0].squeeze(0),
              "cu_seqlens":data_item[1].squeeze(0),
              "image_grid_thw":data_item[2].squeeze(0),               
            }
    # for key, value in inputs.items():
    #     shape_str = str(list(value.shape))
        # print(f"Name: {key:15} | Shape: {shape_str:25} | Type: {value.dtype}")
    return inputs

def get_model_size(ir_path: str, m_type: str = "Mb", verbose: bool = True) -> float:
    xml_size = os.path.getsize(ir_path)
    bin_size = os.path.getsize(os.path.splitext(ir_path)[0] + ".bin")
    for t in ["bytes", "Kb", "Mb"]:
        if m_type == t:
            break
        xml_size /= 1024
        bin_size /= 1024
    model_size = xml_size + bin_size
    if verbose:
        print(f"Model graph (xml):   {xml_size:.3f} Mb")
        print(f"Model weights (bin): {bin_size:.3f} Mb")
        print(f"Model size:          {model_size:.3f} Mb")
    return model_size

path_to_model = "/home/benchmark/xkd/XiaoMi/PaddleOCR-VL/paddleocr_vl_ov/ov-PaddleOCR-VL-model/vision.xml"
int8_ir_save_path = path_to_model.replace(".xml","_int8_q.xml")
ov_model = ov.Core().read_model(path_to_model)

data_count = 10
custom_dataset = CustomDataset(data_count)
val_data_loader = torch.utils.data.DataLoader(custom_dataset, batch_size=1, shuffle=False)
calibration_dataset = nncf.Dataset(val_data_loader, transform_fn)

ov_quantized_model = nncf.quantize(ov_model, calibration_dataset,
                                    preset=nncf.QuantizationPreset.PERFORMANCE,
                                    model_type=nncf.ModelType.TRANSFORMER,
                                    advanced_parameters=nncf.AdvancedQuantizationParameters(
                                    smooth_quant_alphas=nncf.AdvancedSmoothQuantParameters(matmul=-1)
                                ),)

ov.save_model(ov_quantized_model, int8_ir_save_path)