from models.quant import *
from models.layers import norm

def AQD_update(model, args):

    for name, param in model.named_parameters():
        if hasattr(args.quantizer, 'keyword'):
            if 'first-last' in args.quantizer.keyword and name == 'conv1.weight':
                first_quant_conv = quant_conv(param.shape[0], param.shape[1], kernel_size=param.shape[2], args=args)
                param.data.copy_(first_quant_conv(param.data)) 
            elif name != "conv1.weight" and ("conv1.weight" in name or "conv2.weight" in name):
                layer_quant_conv = quant_conv(param.shape[0], param.shape[1], kernel_size=param.shape[2], args=args)
                param.data.copy_(layer_quant_conv(param.data))
            elif "downsample.0.weight" in name:
                quant_conv1x1 = quant_conv(param.shape[0], param.shape[1], kernel_size=1, args=args)
                param.data.copy_(quant_conv1x1(param.data))
            elif 'first-last' in args.quantizer.keyword and name == 'fc.weight':
                last_quant_linear = quant_linear(args.model.last_feature_dim, args.num_classes, bias=True, args=args)
                param.data.copy_(last_quant_linear(param.data))


class WSQConv2d(nn.Conv2d):
    bit1 = [0.7979]
    bit2 = [0.5288, 0.9816]
    bit3 = [0.4510, 0.7481, 0.9882]
    bit4 = [0.2960, 0.5567, 0.7088, 1.1286]
    bit8 = [0.05, 0.1, 0.2, 0.3375, 0.5250, 0.9875, 1.3875, 1.45]

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, rho=1e-3, n_bits=1, clip_prob=-1):
        super(WSQConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        
        self.alpha = torch.tensor(getattr(self, f'bit{n_bits}'), dtype=torch.float32)
        self.rho = rho
        self.clip_prob = clip_prob
        
        # Generate all combinations of b_k in {-1, 1} for 2^(M-1) terms
        b_combinations = torch.cartesian_prod(*[torch.tensor([-1., 1.]) for _ in range(len(self.alpha))])
        if len(self.alpha) == 1:
            b_combinations = b_combinations.unsqueeze(-1)
        q_values = torch.sum(b_combinations * self.alpha, dim=1)
        self.q_values = torch.sort(q_values).values
        self.edges = 0.5 * (self.q_values[1:] + self.q_values[:-1])
        
    def clip_by_prob(self, x):

        original_shape = x.shape
        x_flat = x.view(x.size(0), -1)
        abs_x_flat = x_flat.abs()
        topk = max(1, int(self.clip_prob * abs_x_flat.size(1)))
        thresholds = torch.topk(abs_x_flat, topk, dim=1, largest=True, sorted=True).values[:, -1].view(-1, 1)

        # Clip values in parallel
        clipped_x_flat = torch.where(
            x_flat > thresholds, thresholds,
            torch.where(x_flat < -thresholds, -thresholds, x_flat)
        )

        # Reshape back to the original shape
        clipped_x = clipped_x_flat.view(original_shape)

        return clipped_x
    
    def forward(self, x):
        with torch.no_grad():
            if self.clip_prob > 0:
                x = self.clip_by_prob(x)
            x_mean = x.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            x = x - x_mean
            x_std = x.view(x.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-8
            x = x / x_std.expand_as(x)

            indices = torch.bucketize(x, self.edges, right=False)
            quantized_x = self.q_values[indices]
            quantized_x = quantized_x * x_std + x_mean
        return quantized_x
        

def WSQ_update(model, args):
    
    for name, param in model.named_parameters():
        if hasattr(args.quantizer, 'keyword'):
            if 'first-last' in args.quantizer.keyword and name == 'conv1.weight':
                first_quant_conv = WSQConv2d(param.shape[1], param.shape[0], kernel_size=param.shape[2], 
                                             n_bits=args.quantizer.wt_bit, clip_prob=args.quantizer.wt_clip_prob)
                param.data.copy_(first_quant_conv(param.data)) 
            elif name != "conv1.weight" and ("conv1.weight" in name or "conv2.weight" in name):
                layer_quant_conv = WSQConv2d(param.shape[1], param.shape[0], kernel_size=param.shape[2], 
                                             n_bits=args.quantizer.wt_bit, clip_prob=args.quantizer.wt_clip_prob)
                param.data.copy_(layer_quant_conv(param.data))
            elif "downsample.0.weight" in name:
                quant_conv1x1 = WSQConv2d(param.shape[1], param.shape[0], kernel_size=1, 
                                          n_bits=args.quantizer.wt_bit, clip_prob=args.quantizer.wt_clip_prob)
                param.data.copy_(quant_conv1x1(param.data))
            elif 'first-last' in args.quantizer.keyword and name == 'fc.weight':
                last_quant_linear = quant_linear(args.model.last_feature_dim, args.num_classes, bias=True, 
                                                 n_bits=args.quantizer.wt_bit, clip_prob=args.quantizer.wt_clip_prob)
                param.data.copy_(last_quant_linear(param.data))

                
class WLQConv2d(nn.Conv2d):
    bit1 = [0.7979]
    bit2 = [0.5288, 0.9816]
    bit3 = [0.4510, 0.7481, 0.9882]
    bit4 = [0.2960, 0.5567, 0.7088, 1.1286]
    bit8 = [0.05, 0.1, 0.2, 0.3375, 0.5250, 0.9875, 1.3875, 1.45]

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, rho=1e-3, n_bits=1):
        super(WLQConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        
        self.alpha = torch.tensor(getattr(self, f'bit{n_bits}'), dtype=torch.float32)
        self.rho = rho

        # Generate all combinations of b_k in {-1, 1} for 2^(M-1) terms
        b_combinations = torch.cartesian_prod(*[torch.tensor([-1., 1.]) for _ in range(len(self.alpha))])
        if len(self.alpha) == 1:
            b_combinations = b_combinations.unsqueeze(-1)
        q_values = torch.sum(b_combinations * self.alpha, dim=1)
        self.q_values = torch.sort(q_values).values
        self.edges = 0.5 * (self.q_values[1:] + self.q_values[:-1])
        
    def forward(self, x):
        with torch.no_grad():
            x_mean = x.mean().view(-1, 1, 1, 1)
            x = x - x_mean
            x_std = x.view(-1).std().view(-1, 1, 1, 1) + 1e-8
            x = x / x_std.expand_as(x)

            indices = torch.bucketize(x, self.edges, right=False)
            quantized_x = self.q_values[indices]
            quantized_x = quantized_x * x_std + x_mean
        return quantized_x
        

def WLQ_update(model, args):
    
    for name, param in model.named_parameters():
        if hasattr(args.quantizer, 'keyword'):
            if 'first-last' in args.quantizer.keyword and name == 'conv1.weight':
                first_quant_conv = WLQConv2d(param.shape[1], param.shape[0], kernel_size=param.shape[2], n_bits=args.quantizer.wt_bit)
                param.data.copy_(first_quant_conv(param.data)) 
            elif name != "conv1.weight" and ("conv1.weight" in name or "conv2.weight" in name):
                layer_quant_conv = WLQConv2d(param.shape[1], param.shape[0], kernel_size=param.shape[2], n_bits=args.quantizer.wt_bit)
                param.data.copy_(layer_quant_conv(param.data))
            elif "downsample.0.weight" in name:
                quant_conv1x1 = WLQConv2d(param.shape[1], param.shape[0], kernel_size=1, n_bits=args.quantizer.wt_bit)
                param.data.copy_(quant_conv1x1(param.data))
            elif 'first-last' in args.quantizer.keyword and name == 'fc.weight':
                last_quant_linear = WLQConv2d(args.model.last_feature_dim, args.num_classes, bias=True, n_bits=args.quantizer.wt_bit)
                param.data.copy_(last_quant_linear(param.data))
                
                
def quantize_and_dequantize(tensor, global_tensor, bit_width, lr=1.0):
    residual = tensor - global_tensor
    original_shape = residual.shape
    residual_flatten = residual.view(-1)

    norm = torch.norm(residual_flatten)
    if norm < 1e-12:
        return torch.zeros_like(residual_flatten), torch.zeros_like(residual), global_tensor.clone()

    abs_ratio = torch.abs(residual_flatten) / norm
    levels = 2 ** bit_width

    lower_level = torch.floor(abs_ratio * levels)
    upper_level = torch.ceil(abs_ratio * levels)

    p_upper = (abs_ratio * levels) - lower_level
    random_values = torch.rand_like(residual_flatten)
    selected_levels = torch.where(random_values < p_upper, upper_level, lower_level)

    quantized_flatten = norm * torch.sign(residual_flatten) * (selected_levels / levels)
    
    dequantized_flatten = norm * torch.sign(residual_flatten) * (selected_levels / levels)

    dequantized_tensor = dequantized_flatten.view(original_shape)
    
    updated_global_tensor = global_tensor + lr * dequantized_tensor # lr 어떻게 설정할지.. 일단 1로

    return quantized_flatten, dequantized_tensor, updated_global_tensor

def PAQ_update(model, global_model, args):
    s = args.quantizer.wt_bit

    lr = getattr(args, 'global_PAQ_lr', 1.0)

    g_params = dict(global_model.named_parameters())

    for name, param in model.named_parameters():
        if 'first-last' in args.quantizer.keyword and name == 'conv1.weight':
      
            global_param = g_params[name]
            q, dq, updated_global = quantize_and_dequantize(param.data, global_param.data, s, lr)
            global_param.data.copy_(updated_global)

        elif 'first-last' in args.quantizer.keyword and name == 'fc.weight':
      
            global_param = g_params[name]
            q, dq, updated_global = quantize_and_dequantize(param.data, global_param.data, s, lr)
            global_param.data.copy_(updated_global)

        elif "conv" in name or "downsample.0.weight" in name:
         
            global_param = g_params[name]
            q, dq, updated_global = quantize_and_dequantize(param.data, global_param.data, s, lr)
            global_param.data.copy_(updated_global)
            # print(torch.sum(~(updated_global != g_params[name])).item())

    return global_model
