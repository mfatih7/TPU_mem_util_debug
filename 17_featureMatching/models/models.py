from models.models_exp2 import get_model as model_exp2_get_model

import torch
from torchsummary import summary
from thop import profile
from models.torchSummaryWrapper import get_torchSummaryWrapper
        
def get_model( config, model_type, N, model_width, en_checkpointing ):   
    
    if( N == 512 or N == 1024 or N == 2048 or N == 4096 ):        
        
        if( model_type == 'model_exp2' ):
            return model_exp2_get_model( config, N, model_width, en_checkpointing )
        
        else:
            raise ValueError(f"The provided argument is not valid: {model_type}")
    else:        
        raise ValueError(f"The provided argument is not valid: {N}")

# MobileNetV1 Deptwise Seperable Connection implemented
# MobileNetV1 Width Multiplier and Resolution Multiplier not implemented.
# MobileNetV1 Width Multiplier can be used when constructing networks with same number of parameters.

# MobileNetV2 Inverted Residual with Linear Bottleneck
# MobileNetV2 ReLu6
# MobileNetV2 Shortcut connections when stride is 1 and in channels is equal to out channels

# MobileNetV3 Squeeze and excite
# MobileNetV3 Platform-Aware NAS for Block-wise search
# MobileNetV3 NetAdapt for Layer-wise Search
# MobileNetV3 Redesigning Expensive Layers
# MobileNetV3 h-swish nonlinearity
# MobileNetV3 Change the size of squeeze-and excite-bottleneck(1/4 of the number of channels in expansion layer)

# EfficientNetV1 Compound Scaling ( depth(number of layers), width(number of channels), resolution(input size) )
## EfficientNetV1 Finding constants for compound scaling
## EfficientNetV1 Scaling with respect to the constants
# EfficientNetV1 Stochastic Depth

# EfficientNetV2 Progressively increasing input size during training and adding more regularization as input size increases
# EfficientNetV2 Fused-MBConv


def get_model_structure( config, device, model, N, model_width, en_grad_checkpointing):   
    
    if(en_grad_checkpointing==False):
        summary(model, (config.input_channel_count, N, model_width), batch_size=2, device=device ) # batch_size must be at least 2 to prevent batch norm errors
    else:                    
        summary(get_torchSummaryWrapper( model ), (config.input_channel_count, N, model_width), batch_size=2, device=device ) # batch_size must be at least 2 to prevent batch norm errors
    input_thop = torch.randn(2, config.input_channel_count, N, model_width, device=device)  # Example input tensor
    flops, params = profile(model, inputs=(input_thop, ))
    flops = int(flops)
    params = int(params)
    print(f"Model FLOPs: {flops:,}")
    print(f"Model Parameters: {params:,}")
    
def get_model_params_and_FLOPS( config, device, model, N, model_width, en_grad_checkpointing):   
    input_thop = torch.randn(2, config.input_channel_count, N, model_width, device=device)  # Example input tensor
    flops, params = profile(model, inputs=(input_thop, ))
    flops = int(flops)
    params = int(params)
    print(f"Model FLOPs: {flops:,}")
    print(f"Model Parameters: {params:,}")
    
    return params, flops
    