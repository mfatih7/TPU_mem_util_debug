import torch
import torch.nn as nn
import math

class Non_Lin(nn.Module):
    def __init__(self, non_lin):
        super(Non_Lin, self).__init__()
        
        if( non_lin == 'ReLU' ):
            self.non_lin = nn.ReLU()
        elif( non_lin == 'LeakyReLU' ):
            self.non_lin = nn.LeakyReLU()
        elif non_lin == 'tanh':
            self.non_lin = nn.Tanh()

    def forward(self, x):        
        out = self.non_lin( x )        
        return out

class Context_Norm_1_to_1(nn.Module):   # Makes training 10% slower
    def __init__(self, eps=1e-5):
        super(Context_Norm_1_to_1, self).__init__()
        self.eps = eps

    def forward(self, activation_map):
        mean = activation_map.mean(dim=0, keepdim=True)
        std = activation_map.std(dim=0, keepdim=True) + self.eps

        normalized_map = (activation_map - mean) / std
        return normalized_map

class Conv2d_N(nn.Module):
    def __init__(self, in_channels, out_channels, height, kernel_size, stride, bias, enable_context_norm ):
        super(Conv2d_N, self).__init__()
        
        self.enable_context_norm = enable_context_norm
        
        self.cnn = nn.Conv2d( in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, bias = bias, )
        
        self.norm = nn.BatchNorm2d(out_channels, track_running_stats=False)
            
        if(self.enable_context_norm):
            if( height/int(kernel_size[0]) > 1 and kernel_size[1]==1 ):
                # self.cont_norm = nn.InstanceNorm2d(out_channels, eps=1e-3)  # does not hel for 1 to 1 training
                self.cont_norm = Context_Norm_1_to_1()
            else:
                self.cont_norm = nn.Identity()
            
    def forward(self, x):
        if(self.enable_context_norm):
            x = self.norm( self.cont_norm( self.cnn(x) ) )
        else:
            x = self.norm( self.cnn(x) )
        return x
    
class Width_Reduction(nn.Module):
    def __init__(self, in_width, out_channels, height, enable_context_norm, non_lin):
        super(Width_Reduction, self).__init__()
        
        self.width_reduction = Conv2d_N( in_channels = 2, out_channels = out_channels, height = height, kernel_size = (1, in_width), stride = (1,1), bias = False, enable_context_norm = enable_context_norm )
        
        self.non_lin = Non_Lin( non_lin )
            
    def forward(self, x):
        x = self.non_lin( self.width_reduction(x) )
        return x
    
class Height_Reduction(nn.Module):
    def __init__(self, in_channels, out_channels, height, enable_context_norm, non_lin):
        super(Height_Reduction, self).__init__()
        
        self.height_reduction = Conv2d_N( in_channels = in_channels, out_channels = out_channels, height = height, kernel_size = (2,1), stride = (2,1), bias = False, enable_context_norm = enable_context_norm )
        
        self.non_lin = Non_Lin( non_lin )
            
    def forward(self, x):
        x = self.non_lin( self.height_reduction(x) )
        return x
    
class Channel_Reduction(nn.Module):
    def __init__(self, in_channels, out_channels, height, enable_context_norm, non_lin):
        super(Channel_Reduction, self).__init__()
        
        self.channel_reduction = Conv2d_N( in_channels = in_channels, out_channels = out_channels, height = height, kernel_size = (1,1), stride = (1,1), bias = False, enable_context_norm = enable_context_norm )
        
        self.non_lin = Non_Lin( non_lin )
            
    def forward(self, x):
        x = self.non_lin( self.channel_reduction(x) )
        return x
    
class Pointwise_Conv_Shortcut(nn.Module):
    def __init__(self, channels, height, enable_context_norm, non_lin):
        super(Pointwise_Conv_Shortcut, self).__init__()
        
        self.pointwise_conv = Conv2d_N( in_channels = channels, out_channels = channels, height = height, kernel_size = (1,1), stride = (1,1), bias = False, enable_context_norm = enable_context_norm )
        
        self.non_lin = Non_Lin( non_lin )
            
    def forward(self, x):
        shortcut = x
        x = self.pointwise_conv(x)
        x = self.non_lin( x )
        x = shortcut + x        
        return x
    
class Depth_Wise_Seperable_Convolution(nn.Module):
    def __init__(self, in_channels, out_channels, enable_context_norm, non_lin):
        super(Depth_Wise_Seperable_Convolution, self).__init__()
        
        self.depth_wise_conv = nn.Conv2d( in_channels = in_channels, out_channels = in_channels, kernel_size = (2,1), stride = (2,1), groups = in_channels, bias = False )
        
        ## enable_context_norm can be added
                
        self.norm_1 = nn.BatchNorm2d( in_channels, track_running_stats=False )
        
        self.point_wise_conv = nn.Conv2d( in_channels = in_channels, out_channels = out_channels, kernel_size = (1,1), stride = (1,1), bias = False )
        
        ## enable_context_norm can be added
        
        self.norm_2 = nn.BatchNorm2d( out_channels, track_running_stats=False )
            
        self.non_lin = Non_Lin( non_lin )
            
    def forward(self, x):        
        
        x = self.non_lin( self.norm_1( self.depth_wise_conv( x ) ) )
        x = self.non_lin( self.norm_2( self.point_wise_conv( x ) ) )
        return x
    
class Pool_1_to_1(nn.Module):
    def __init__(self, in_channels, out_channels, ):
        super(Pool_1_to_1, self).__init__()
        
        self.conv = nn.Conv2d( in_channels = in_channels, out_channels = out_channels, kernel_size = (1,1), stride = (1,1), bias = False, )        
        self.softmax = nn.Softmax(dim=2)    
        
    def forward(self, x):
        
        out = self.conv(x)
        Spool = self.softmax(out)        
        out = torch.matmul( x.squeeze(3), torch.transpose(Spool, 1, 2).squeeze(3) ).unsqueeze(3)        
        return out
    
class Unpool_1_to_1(nn.Module):
    def __init__(self, in_channels, out_channels, ):
        super(Unpool_1_to_1, self).__init__()
        
        self.conv = nn.Conv2d( in_channels=in_channels, out_channels=out_channels, kernel_size=(1,1), stride=(1,1) ) 
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x_0, x_1):
        
        out = self.conv(x_0)
        Sunpool = self.softmax(out)        
        out = torch.matmul( x_1.squeeze(3), Sunpool.squeeze(3) ).unsqueeze(3)        
        return out
    
class Order_Aware_Filtering_1_to_1(nn.Module):
    def __init__(self, in_channels, non_lin):
        super(Order_Aware_Filtering_1_to_1, self).__init__()
        
        self.conv = nn.Conv2d( in_channels = in_channels, out_channels = in_channels, kernel_size = (1,1), stride = (1,1), bias = False, )
        
        self.norm = nn.BatchNorm2d(in_channels, track_running_stats=False)
        
        self.non_lin = Non_Lin( non_lin )
        
    def forward(self, x):
        
        out = self.norm( self.conv(x) )
        
        out = self.non_lin( out + x )   # Shortcut connections
             
        return out
    
class Block_Order_Aware_Filtering_1_to_1(nn.Module):
    def __init__(self, N, in_channels, height, n_layers, non_lin):
        super(Block_Order_Aware_Filtering_1_to_1, self).__init__()
        
        self.M = int( N/8 )
        
        self.pool_1_to_1 = Pool_1_to_1( in_channels = in_channels, out_channels = self.M, )
        
        order_Aware_Filtering_1_to_1_layers = []
        
        for lay in range( n_layers ):
            order_Aware_Filtering_1_to_1_layers.append( Order_Aware_Filtering_1_to_1( in_channels = self.M, non_lin = non_lin, ) )
            
        self.order_Aware_Filtering_1_to_1_net = nn.Sequential(*order_Aware_Filtering_1_to_1_layers)
        
        self.unpool_1_to_1 = Unpool_1_to_1( in_channels = in_channels, out_channels = self.M, )
        
    def forward(self, x):
        
        # Shortcut connection is on the block
        
        out = self.pool_1_to_1(x)
        
        out = torch.transpose(out, 1, 2)
        
        out = self.order_Aware_Filtering_1_to_1_net(out)
        
        out = torch.transpose(out, 1, 2)
        
        out = self.unpool_1_to_1(x, out)
             
        return out

class Block(nn.Module):
    def __init__(self, N,
                       block_no,
                       in_width,
                       channels_0,
                       channels_1,
                       channels_2,
                       height,
                       height_reduction_type,
                       channel_reduction_layer_enable,
                       enable_context_norm,
                       non_lin,
                       pointwise_conv_count,
                       order_aware_filter_count, ):
        super(Block, self).__init__()  
                
        self.block_no = block_no
        self.channel_reduction_layer_enable = channel_reduction_layer_enable
        self.pointwise_conv_count = pointwise_conv_count
        self.order_aware_filter_count = order_aware_filter_count
        
        if(block_no==0):            
            self.width_reduction = Width_Reduction(   in_width = in_width,
                                                      out_channels = channels_1,
                                                      height = height,
                                                      enable_context_norm = enable_context_norm,
                                                      non_lin = non_lin )
        else:
            
            if(height_reduction_type=='normal_conv'):
            
                self.height_reduction = Height_Reduction( in_channels = channels_0,
                                                          out_channels = channels_1,
                                                          height = height,
                                                          enable_context_norm = enable_context_norm,
                                                          non_lin = non_lin )
            elif(height_reduction_type=='depth_wise_sep_conv'):
            
                self.height_reduction = Depth_Wise_Seperable_Convolution( in_channels = channels_0,
                                                                          out_channels = channels_1,
                                                                          enable_context_norm = enable_context_norm,
                                                                          non_lin = non_lin )
            height = int( height/2 )
        
        if(channel_reduction_layer_enable):
            self.channel_reduction = Channel_Reduction( in_channels = channels_1, 
                                                        out_channels = channels_2,
                                                        height = height,
                                                        enable_context_norm = enable_context_norm,
                                                        non_lin = non_lin, )
            
            
        if( self.pointwise_conv_count > 0 ):
            
            pointwise_conv_layers = []
            
            for lay in range( pointwise_conv_count ):
                pointwise_conv_layers.append( Pointwise_Conv_Shortcut( channels = channels_2,
                                                                       height = height,
                                                                       enable_context_norm = enable_context_norm,
                                                                       non_lin = non_lin ) )
                
            self.pointwise_conv_layers_net = nn.Sequential(*pointwise_conv_layers)            
            
        if( self.order_aware_filter_count > 0 ):
            
            self.block_Order_Aware_Filtering_1_to_1 = Block_Order_Aware_Filtering_1_to_1( N = N,
                                                                                          in_channels = channels_2,
                                                                                          height = height,
                                                                                          n_layers = self.order_aware_filter_count,
                                                                                          non_lin = non_lin )
        
    def forward(self, x):
        
        if( self.block_no == 0 ):
            out = self.width_reduction( x )
        else:
            out = self.height_reduction( x )
            
        if( self.channel_reduction_layer_enable ):
            out = self.channel_reduction( out )            
            
        if( self.pointwise_conv_count > 0 and self.order_aware_filter_count == 0):
            
            # shortcut connections are on pointwise_conv_layers
            
            out = self.pointwise_conv_layers_net(out)
            
        elif( self.pointwise_conv_count == 0 and self.order_aware_filter_count > 0):
            
            out1 = self.block_Order_Aware_Filtering_1_to_1(out)
            
            out = out1 + out # shortcut connection
            
        elif( self.pointwise_conv_count > 0 and self.order_aware_filter_count > 0):
            
            # shortcut connections are on pointwise_conv_layers
            
            out1 = self.block_Order_Aware_Filtering_1_to_1(out)
            
            out2 = self.pointwise_conv_layers_net(out)
            
            out = out1 + out2
            
        return out
    
class model_exp_00(nn.Module):
    def __init__(self,  N, 
                        in_width,
                        init_channel_count,
                        ch_expans_base_param,
                        ch_expans_power_param,
                        height_reduction_type,
                        channel_reduction_layer_enable,
                        channel_reduction_ratio,
                        pointwise_conv_layers_count,
                        pointwise_conv_layers_power_param,
                        order_aware_filter_layers_count,
                        order_aware_filter_layers_power_param,
                        enable_context_norm,
                        non_lin, ):
        super(model_exp_00, self).__init__()        
        
        self.N = N
        self.in_width = in_width
        self.init_channel_count = init_channel_count
        self.ch_expans_base_param = ch_expans_base_param
        self.ch_expans_power_param = ch_expans_power_param
        self.height_reduction_type = height_reduction_type
        self.channel_reduction_layer_enable = channel_reduction_layer_enable
        self.channel_reduction_ratio = channel_reduction_ratio
        self.pointwise_conv_layers_count = pointwise_conv_layers_count
        self.pointwise_conv_layers_power_param = pointwise_conv_layers_power_param
        self.order_aware_filter_layers_count = order_aware_filter_layers_count
        self.order_aware_filter_layers_power_param = order_aware_filter_layers_power_param
        self.enable_context_norm = enable_context_norm
        self.non_lin = non_lin
        
        self.n_blocks = int( math.log2(N) + 1 )

        height = N
        
        block_in_channel_counts, block_out_channel_counts = self.calculate_block_channel_counts()        
        
        _, pointwise_conv_counts = self.distribute_integer( pointwise_or_order_aware = 'pointwise', N = self.n_blocks, n = self.n_blocks, a = self.pointwise_conv_layers_count, power_param = self.pointwise_conv_layers_power_param )
        _, order_aware_filter_counts = self.distribute_integer( pointwise_or_order_aware = 'order_aware', N = self.n_blocks, n = self.n_blocks-1, a = self.order_aware_filter_layers_count, power_param = self.order_aware_filter_layers_power_param )
        
        layers = []
        
        for block_no in range(self.n_blocks):        

            if(block_no==0):
                channels_0 = 2
            else:
                channels_0 = block_out_channel_counts[block_no-1]
            
            layers.append( Block( N = N,
                                  block_no = block_no,
                                  in_width = self.in_width,
                                  channels_0 = channels_0,
                                  channels_1 = block_in_channel_counts[block_no],
                                  channels_2 = block_out_channel_counts[block_no],
                                  height = height,
                                  height_reduction_type = height_reduction_type,
                                  channel_reduction_layer_enable = channel_reduction_layer_enable,
                                  enable_context_norm = enable_context_norm,
                                  non_lin = non_lin,
                                  pointwise_conv_count = pointwise_conv_counts[block_no],
                                  order_aware_filter_count = order_aware_filter_counts[block_no], ) )
            if(block_no>0):
                height = int( height / 2 ) # Height is reduced at each blocks
            
        self.net = nn.Sequential(*layers)
        
        self.initial_fully_connected_size = block_out_channel_counts[-1]
        
        self.fc1 = nn.Linear(self.initial_fully_connected_size * 1 * 1, int(self.initial_fully_connected_size/4))
        
        self.fc2 = nn.Linear( int(self.initial_fully_connected_size/4) * 1 * 1, 1)
        
        self.non_lin = Non_Lin( non_lin )
            
    def calculate_block_channel_counts(self):
        
        n0_matches = torch.zeros( (self.n_blocks), dtype=torch.int32 )
        n1_info_size = torch.zeros( (self.n_blocks), dtype=torch.int32 )
        n2_info_size_with_channels_raw = torch.zeros( (self.n_blocks), dtype=torch.int32 )
        n3_info_size_with_channels_processed = torch.zeros( (self.n_blocks), dtype=torch.float32 )
        n4_info_size_with_channels_processed_reduced = torch.zeros( (self.n_blocks), dtype=torch.float32 )

        for i in range( self.n_blocks ):
            
            n0_matches[i] = 2**i
            
            for j in range( n0_matches[i], 0, -1 ):
                n1_info_size[i] += j
                
            n2_info_size_with_channels_raw[i] = self.init_channel_count * n1_info_size[i]
            
            n3_info_size_with_channels_processed[i] = n2_info_size_with_channels_raw[i] * (self.ch_expans_base_param**(i*self.ch_expans_power_param))
            
            n4_info_size_with_channels_processed_reduced[i] = n3_info_size_with_channels_processed[i] * self.channel_reduction_ratio
        
        return n3_info_size_with_channels_processed.to(torch.int), n4_info_size_with_channels_processed_reduced.to(torch.int)
    
    def distribute_integer(self, pointwise_or_order_aware, N, n, a, power_param):
        """
        Distributes the integer 'a' among elements based on the given weights.
        Ensures that each element gets an integer value.

        :param a: Integer to be distributed.
        :param weights: List of float weights for each element.
        :return: List of integers representing the distributed amounts.
        """
        
        weights = []
        
        if( pointwise_or_order_aware == 'pointwise' ):
            for i in range(N):
                weights.append(i**power_param)
        elif( pointwise_or_order_aware == 'order_aware' ):
            for i in range(N):
                if(i%3==2):
                    weights.append(i**power_param)
                else:
                    weights.append(0)                    

        # Normalize weights so their sum is 1
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        # Initial distribution based on weights
        distribution = [int(a * w) for w in normalized_weights]

        # Adjust for rounding errors
        distributed_sum = sum(distribution)
        difference = a - distributed_sum

        # Sorting indices based on the fractional part lost during rounding
        fractional_parts = sorted(range(n), key=lambda i: normalized_weights[i] * a - distribution[i], reverse=True)

        # Distributing the remaining amount based on the fractional parts
        for i in range(difference):
            distribution[fractional_parts[i]] += 1

        return weights, distribution
    
    def forward(self, x):
        
        x = self.net( x )
        
        x = x.view( -1, self.initial_fully_connected_size * 1 * 1 )
        x = self.non_lin( self.fc1(x) )
        x = self.fc2(x)
        
        return x
        
def get_model( config, N, model_width, en_checkpointing, model_adjust_params = None ):   

    if( N == 512 or N == 1024 or N == 2048 ):
        
        in_width = model_width
        
        non_lin = 'ReLU'
        # non_lin = 'LeakyReLU'  
        # non_lin = 'tanh'        
        
        if( config.model_exp_no < 10000 ):
            enable_context_norm = False
        else:
            enable_context_norm = True
        
        if(model_adjust_params != None):            
            init_channel_count = model_adjust_params[0]
            
            ch_expans_base_param = model_adjust_params[1]
            ch_expans_power_param = model_adjust_params[2]
            
            height_reduction_type = model_adjust_params[3]
            
            channel_reduction_layer_enable = model_adjust_params[4]
            channel_reduction_ratio = model_adjust_params[5]
            
            pointwise_conv_layers_count = model_adjust_params[6]
            pointwise_conv_layers_power_param = model_adjust_params[7]

            order_aware_filter_layers_count = model_adjust_params[8]
            order_aware_filter_layers_power_param = model_adjust_params[9]
        else:        
            
            if( (config.model_exp_no >= 0 and config.model_exp_no < 4 ) or (config.model_exp_no >= 10000 and config.model_exp_no < 10004 ) ):
                
                pointwise_conv_layers_counts = [0, 1, 0, 1]
                order_aware_filter_layers_counts = [0, 0, 0.55, 0.55]
                
                init_channel_count = 64
                
                ch_expans_base_params = [[0.3469962594, 0.3357526717, 0.3450461309, 0.3341902633],
                                         [0.3480141413, 0.3380825185, 0.345793587, 0.336181442],
                                         [0.348712847, 0.339827474, 0.3434041153, 0.3354239051]]
                
                if( config.model_exp_no < 10000 ):
                    exp_no_list_index = config.model_exp_no - 0
                else:
                    exp_no_list_index = config.model_exp_no - 10000
                
                ch_expans_base_param = ch_expans_base_params[ int( math.log2(N) - math.log2(512) ) ][exp_no_list_index]                
                # ch_expans_power_param = 0.9
                ch_expans_power_param = 1
                
                height_reduction_type = 'normal_conv'
                channel_reduction_layer_enable = 0
                channel_reduction_ratio = 1
                
                pointwise_conv_layers_count = int( pointwise_conv_layers_counts[exp_no_list_index] * ( math.log2(N) + 1 ) )
                # pointwise_conv_layers_count = 0
                pointwise_conv_layers_power_param = 0
                
                order_aware_filter_layers_count = int( order_aware_filter_layers_counts[exp_no_list_index] * math.log2(N) )
                # order_aware_filter_layers_count = 0
                order_aware_filter_layers_power_param = 0
                
            elif( (config.model_exp_no >= 10 and config.model_exp_no <14 ) or (config.model_exp_no >= 10010 and config.model_exp_no < 10014 ) ):
                
                pointwise_conv_layers_counts = [0, 1, 0, 1]
                order_aware_filter_layers_counts = [0, 0, 1, 1]
                
                init_channel_count = 64
                
                ch_expans_base_params = [[0.3469962594, 0.3357526717, 0.3446162366, 0.333756002],
                                         [0.3480141413, 0.3380825185, 0.3450670316, 0.335505157],
                                         [0.348712847, 0.339827474, 0.3418968305, 0.333999026]]
                
                if( config.model_exp_no < 10000 ):
                    exp_no_list_index = config.model_exp_no - 10
                else:
                    exp_no_list_index = config.model_exp_no - 10010
                
                ch_expans_base_param = ch_expans_base_params[ int( math.log2(N) - math.log2(512) ) ][exp_no_list_index]                
                # ch_expans_power_param = 0.9
                ch_expans_power_param = 1
                
                height_reduction_type = 'normal_conv'
                channel_reduction_layer_enable = 0
                channel_reduction_ratio = 1
                
                pointwise_conv_layers_count = int( pointwise_conv_layers_counts[exp_no_list_index] * ( math.log2(N) + 1 ) )
                # pointwise_conv_layers_count = 0
                pointwise_conv_layers_power_param = 0
                
                order_aware_filter_layers_count = int( order_aware_filter_layers_counts[exp_no_list_index] * math.log2(N) )
                # order_aware_filter_layers_count = 0
                order_aware_filter_layers_power_param = 0
            
            else:
                raise ValueError(f"The provided argument is not valid: {config.model_exp_no}")
                
        return model_exp_00( N, 
                             in_width,
                             init_channel_count,
                             ch_expans_base_param,
                             ch_expans_power_param,
                             height_reduction_type,
                             channel_reduction_layer_enable,
                             channel_reduction_ratio,
                             pointwise_conv_layers_count,
                             pointwise_conv_layers_power_param,
                             order_aware_filter_layers_count,
                             order_aware_filter_layers_power_param,
                             enable_context_norm,
                             non_lin, )
    else:
        raise ValueError(f"The provided argument is not valid: {N}")

if __name__ == '__main__':
    
    import os    
    os.chdir( os.path.dirname( os.getcwd( ) ) )       
    from config import get_config
    from models.models import get_model_structure
    from models.models import get_model_params_and_FLOPS
    config = get_config()
    
    device = 'cpu'
    
    # N = 512
    # N = 1024
    N = 2048
    model_width = 4
    en_checkpointing = False
    
####################################################################################
    
    first_model_no = 0
    last_model_no = 4
    
    # first_model_no = 10
    # last_model_no = 14
        
####################################################################################
    
    # first_model_no = 10000
    # last_model_no = 10004
    
    # first_model_no = 10010
    # last_model_no = 10014
    
####################################################################################
    
    for i in range( first_model_no, last_model_no, 1 ):
        config.model_exp_no = i 
        print(f'config.model_exp_no: {config.model_exp_no}')
        model = get_model( config, N, model_width, en_checkpointing ).to(device)
        get_model_structure( config, device, model, N, model_width, en_grad_checkpointing=0)
        # get_model_params_and_FLOPS( config, device, model, N, model_width, en_grad_checkpointing=0)
        print('-'*80)

