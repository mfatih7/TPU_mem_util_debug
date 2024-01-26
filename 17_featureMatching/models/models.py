from models.models_CNN_Plain import get_model as CNN_Plain_get_model
from models.models_CNN_Residual import get_model as CNN_Residual_get_model
from models.models_MobileNetV1 import get_model as MobileNetV1_get_model
from models.models_MobileNetV2 import get_model as MobileNetV2_get_model
from models.models_MobileNetV3 import get_model as MobileNetV3_get_model

import torch
from torchsummary import summary
from thop import profile
from models.torchSummaryWrapper import get_torchSummaryWrapper
        
def get_model( model_type, N, model_width, bn_or_gn, en_checkpointing ):   
    
    if( N == 512 or N == 1024 or N == 2048 or N == 4096 ):        
        
        if( model_type == 'CNN_Plain' ):    
            return CNN_Plain_get_model( N, model_width, bn_or_gn, en_checkpointing )    
        elif( model_type == 'CNN_Residual' ):    
            return CNN_Residual_get_model( N, model_width, bn_or_gn, en_checkpointing )
        elif( model_type == 'MobileNetV1' ):
            return MobileNetV1_get_model( N, bn_or_gn, en_checkpointing )    
        elif( model_type == 'MobileNetV2' ):
            return MobileNetV2_get_model( N, bn_or_gn, en_checkpointing )    
        elif( model_type == 'MobileNetV3' ):
            return MobileNetV3_get_model( N, bn_or_gn, en_checkpointing )        
        
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
        summary(model, (config.input_channel_count, N, model_width), batch_size=-1, device=device )
    else:                    
        summary(get_torchSummaryWrapper( model ), (config.input_channel_count, N, model_width), batch_size=-1, device=device )
    input_thop = torch.randn(1, config.input_channel_count, N, model_width, device=device)  # Example input tensor
    flops, params = profile(model, inputs=(input_thop, ))
    print(f"Model FLOPs: {flops}")
    print(f"Model Parameters: {params}")