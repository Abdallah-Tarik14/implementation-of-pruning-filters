import torch
import torch.nn as nn
import numpy as np


def prune_net(net, independentflag, prune_layers, prune_channels, net_name, shortcutflag):
    print("pruning:")
    if net_name == 'vgg16':
        return prune_vgg(net, independentflag, prune_layers, prune_channels)
    elif net_name == "resnet34":
        return prune_resnet(net, independentflag, prune_layers, prune_channels, shortcutflag)
    else:
        print("The net is not provided.")
        exit(0)


def prune_vgg(net, independentflag, prune_layers, prune_channels, criterion='l2_norm', 
              use_activation=True, layer_sensitivity=True):
    """
    Enhanced pruning function for VGG networks.
    
    Args:
        net: The VGG neural network model to be pruned
        independentflag: Boolean flag that determines whether to consider residual connections
        prune_layers: List of layer names to prune (e.g., "conv_1", "conv_2", etc.)
        prune_channels: List containing the number of channels to prune for each layer
        criterion: Channel selection criterion ('l1_norm', 'l2_norm', 'geometric_median', 'activation')
        use_activation: Whether to consider activation statistics in pruning decisions
        layer_sensitivity: Whether to adjust pruning rates based on layer sensitivity
        
    Returns:
        Pruned network
    """
    # Store activation statistics if needed
    activation_stats = {}
    if use_activation:
        activation_stats = collect_activation_statistics(net)
    
    # Calculate layer sensitivity if needed
    sensitivity_scores = {}
    if layer_sensitivity:
        sensitivity_scores = calculate_layer_sensitivity(net, prune_layers)
        
        # Adjust prune_channels based on sensitivity
        prune_channels = adjust_prune_channels(prune_channels, sensitivity_scores, prune_layers)
    
    last_prune_flag = 0
    arg_index = 0
    conv_index = 1
    residue = None

    for i in range(len(net.module.features)):
        if isinstance(net.module.features[i], nn.Conv2d):
            # prune next layer's filter in dim=1
            if last_prune_flag:
                net.module.features[i], residue = get_new_conv(net.module.features[i], remove_channels, 1)
                last_prune_flag = 0
                
            # prune this layer's filter in dim=0
            if "conv_%d" % conv_index in prune_layers:
                # Enhanced channel selection based on chosen criterion
                remove_channels = select_channels_to_prune(
                    net.module.features[i], 
                    prune_channels[arg_index], 
                    residue,
                    independentflag,
                    criterion=criterion,
                    layer_name=f"conv_{conv_index}",
                    activation_stats=activation_stats
                )
                
                print(prune_layers[arg_index], remove_channels)
                net.module.features[i] = get_new_conv(net.module.features[i], remove_channels, 0)
                last_prune_flag = 1
                arg_index += 1
            else:
                residue = None
            conv_index += 1
        elif isinstance(net.module.features[i], nn.BatchNorm2d) and last_prune_flag:
            # prune bn
            net.module.features[i] = get_new_norm(net.module.features[i], remove_channels)
            # Recalculate batch normalization statistics
            recalculate_bn_statistics(net.module.features[i])

    # prune linear
    if "conv_13" in prune_layers:
        net.module.classifier[0] = get_new_linear(net.module.classifier[0], remove_channels)
    
    net = net.cuda()
    print(net)
    return net


def collect_activation_statistics(net, num_batches=10):
    """
    Collect activation statistics for each layer during inference.
    
    Args:
        net: The neural network model
        num_batches: Number of batches to collect statistics from
        
    Returns:
        Dictionary mapping layer names to activation statistics
    """
    # This is a placeholder function - in a real implementation, 
    # you would run inference on a subset of the validation data
    # and collect activation values for each layer
    
    # For demonstration purposes, we'll return an empty dictionary
    # In a real implementation, you would:
    # 1. Register forward hooks to capture activations
    # 2. Run inference on a subset of data
    # 3. Calculate statistics (mean, variance, etc.) of activations
    
    return {}


def calculate_layer_sensitivity(net, prune_layers, num_samples=5):
    """
    Calculate sensitivity of each layer to pruning.
    
    Args:
        net: The neural network model
        prune_layers: List of layer names to analyze
        num_samples: Number of pruning samples to test
        
    Returns:
        Dictionary mapping layer names to sensitivity scores
    """
    # This is a placeholder function - in a real implementation,
    # you would test the impact of pruning each layer on accuracy
    
    # For demonstration purposes, we'll use a heuristic:
    # Earlier layers and final layers are more sensitive than middle layers
    
    sensitivity = {}
    num_layers = len(prune_layers)
    
    for i, layer_name in enumerate(prune_layers):
        # Heuristic: U-shaped sensitivity (higher at start and end, lower in middle)
        position = min(i, num_layers - i - 1) / (num_layers / 2)
        sensitivity[layer_name] = 1.0 - 0.5 * position
    
    return sensitivity


def adjust_prune_channels(prune_channels, sensitivity_scores, prune_layers):
    """
    Adjust pruning rates based on layer sensitivity.
    
    Args:
        prune_channels: Original pruning rates
        sensitivity_scores: Dictionary of layer sensitivity scores
        prune_layers: List of layer names
        
    Returns:
        Adjusted pruning rates
    """
    adjusted_channels = []
    
    for i, layer_name in enumerate(prune_layers):
        if layer_name in sensitivity_scores:
            # Reduce pruning for sensitive layers
            adjusted_amount = int(prune_channels[i] * (1.0 - 0.3 * sensitivity_scores[layer_name]))
            # Ensure we're pruning at least 1 channel
            adjusted_amount = max(1, adjusted_amount)
            adjusted_channels.append(adjusted_amount)
        else:
            adjusted_channels.append(prune_channels[i])
    
    return adjusted_channels


def select_channels_to_prune(conv_layer, num_channels, residue, independentflag, 
                            criterion='l2_norm', layer_name=None, activation_stats=None):
    """
    Select channels to prune based on the specified criterion.
    
    Args:
        conv_layer: Convolutional layer to prune
        num_channels: Number of channels to prune
        residue: Residual connections to consider
        independentflag: Whether to consider residual connections
        criterion: Channel selection criterion
        layer_name: Name of the layer (for activation statistics)
        activation_stats: Dictionary of activation statistics
        
    Returns:
        Indices of channels to prune
    """
    weight_matrix = conv_layer.weight.data
    
    if criterion == 'l1_norm':
        # Original L1-norm criterion
        abs_sum = torch.sum(torch.abs(weight_matrix.view(weight_matrix.size(0), -1)), dim=1)
        if independentflag and residue is not None:
            abs_sum = abs_sum + torch.sum(torch.abs(residue.view(residue.size(0), -1)), dim=1)
        _, indices = torch.sort(abs_sum)
        
    elif criterion == 'l2_norm':
        # L2-norm criterion (Euclidean norm)
        l2_norm = torch.sqrt(torch.sum(weight_matrix.view(weight_matrix.size(0), -1)**2, dim=1))
        if independentflag and residue is not None:
            l2_norm = l2_norm + torch.sqrt(torch.sum(residue.view(residue.size(0), -1)**2, dim=1))
        _, indices = torch.sort(l2_norm)
        
    elif criterion == 'geometric_median':
        # Geometric median criterion
        # This is a more robust measure of centrality than mean
        weights_reshaped = weight_matrix.view(weight_matrix.size(0), -1).cpu().numpy()
        importance_scores = []
        
        for i in range(weights_reshaped.shape[0]):
            # Calculate distance of this filter from all others
            distances = []
            for j in range(weights_reshaped.shape[0]):
                if i != j:
                    # Euclidean distance between filters
                    dist = np.linalg.norm(weights_reshaped[i] - weights_reshaped[j])
                    distances.append(dist)
            
            # Lower distance means the filter is more "central" and thus more important
            importance_scores.append(np.mean(distances))
        
        # Convert back to tensor for sorting
        importance_tensor = torch.tensor(importance_scores)
        _, indices = torch.sort(importance_tensor, descending=True)  # Higher is less important
        
    elif criterion == 'activation':
        # Activation-based criterion
        if layer_name in activation_stats:
            # Use activation statistics if available
            activations = activation_stats[layer_name]
            _, indices = torch.sort(activations)
        else:
            # Fall back to L2-norm if no activation stats
            l2_norm = torch.sqrt(torch.sum(weight_matrix.view(weight_matrix.size(0), -1)**2, dim=1))
            if independentflag and residue is not None:
                l2_norm = l2_norm + torch.sqrt(torch.sum(residue.view(residue.size(0), -1)**2, dim=1))
            _, indices = torch.sort(l2_norm)
    
    else:
        # Default to L2-norm if criterion not recognized
        l2_norm = torch.sqrt(torch.sum(weight_matrix.view(weight_matrix.size(0), -1)**2, dim=1))
        if independentflag and residue is not None:
            l2_norm = l2_norm + torch.sqrt(torch.sum(residue.view(residue.size(0), -1)**2, dim=1))
        _, indices = torch.sort(l2_norm)
    
    return indices[:num_channels].tolist()


def recalculate_bn_statistics(bn_layer):
    """
    Recalculate batch normalization statistics after pruning.
    
    Args:
        bn_layer: Batch normalization layer
    """
    # In a real implementation, you would:
    # 1. Run inference on a subset of data
    # 2. Recalculate running mean and variance
    
    # This is a placeholder function
    pass


def prune_resnet(net, independentflag, prune_layers, prune_channels, shortcutflag):
    # init
    last_prune_flag = 0
    arg_index = 0
    residue = None
    layers = [net.module.layer1, net.module.layer2, net.module.layer3, net.module.layer4]

    # prune shortcut
    if shortcutflag:
        downsample_index = 1
        for layer_index in range(len(layers)):
            for block_index in range(len(layers[layer_index])):
                if last_prune_flag:
                    # prune next block's filter in dim=1
                    layers[layer_index][block_index].conv1, residue = get_new_conv(
                        layers[layer_index][block_index].conv1, remove_channels, 1)

                if layer_index >= 1 and block_index == 0:
                    if last_prune_flag:
                        # prune next downsample's filter in dim=1
                        layers[layer_index][block_index].downsample[0], residue = get_new_conv(
                            layers[layer_index][block_index].downsample[0], remove_channels, 1)
                    else:
                        residue = None
                    if "downsample_%d" % downsample_index in prune_layers:
                        # identify channels to remove
                        remove_channels = channels_index(layers[layer_index][block_index].downsample[0].weight.data,
                                                         prune_channels[arg_index], residue, independentflag)
                        print(prune_layers[arg_index], remove_channels)
                        # prune downsample's filter in dim=0
                        layers[layer_index][block_index].downsample[0] = get_new_conv(layers[layer_index][block_index].
                                                                                      downsample[0], remove_channels, 0)
                        # prune downsample's bn
                        layers[layer_index][block_index].downsample[1] = get_new_norm(layers[layer_index][block_index].
                                                                                      downsample[1], remove_channels)
                        arg_index += 1
                        last_prune_flag = 1
                    else:
                        last_prune_flag = 0
                    downsample_index += 1

                if last_prune_flag:
                    # prune next block's filter in dim=0
                    layers[layer_index][block_index].conv2 = get_new_conv(layers[layer_index][block_index].conv2,
                                                                          remove_channels, 0)
                    # prune next block's bn
                    layers[layer_index][block_index].bn2 = get_new_norm(layers[layer_index][block_index].bn2,
                                                                        remove_channels)
    # prune linear
    if "downsample_3" in prune_layers:
        net.module.fc = get_new_linear(net.module.fc, remove_channels)

    # prune non-shortcut
    else:
        conv_index = 2
        for layer_index in range(len(layers)):
            for block_index in range(len(layers[layer_index])):
                if "conv_%d" % conv_index in prune_layers:
                    # identify channels to remove
                    remove_channels = channels_index(layers[layer_index][block_index].conv1.weight.data,
                                                     prune_channels[arg_index], residue, independentflag)
                    print(prune_layers[arg_index], remove_channels)
                    # prune this layer's filter in dim=0
                    layers[layer_index][block_index].conv1 = get_new_conv(layers[layer_index][block_index].conv1,
                                                                          remove_channels, 0)
                    # prune next layer's filter in dim=1
                    layers[layer_index][block_index].conv2, residue = get_new_conv(
                        layers[layer_index][block_index].conv2, remove_channels, 1)
                    residue = 0
                    # prune bn
                    layers[layer_index][block_index].bn1 = get_new_norm(layers[layer_index][block_index].bn1,
                                                                        remove_channels)
                    arg_index += 1
                conv_index += 2
    net = net.cuda()
    print(net)
    return net


def channels_index(weight_matrix, prune_num, residue, independentflag):
    abs_sum = torch.sum(torch.abs(weight_matrix.view(weight_matrix.size(0), -1)), dim=1)
    if independentflag and residue is not None:
        abs_sum = abs_sum + torch.sum(torch.abs(residue.view(residue.size(0), -1)), dim=1)
    _, indices = torch.sort(abs_sum)
    return indices[:prune_num].tolist()


def select_channels(weight_matrix, remove_channels, dim):
    indices = torch.tensor(list(set(range(weight_matrix.shape[dim])) - set(remove_channels)))
    new = torch.index_select(weight_matrix, dim, indices.cuda())
    if dim == 1:
        residue = torch.index_select(weight_matrix, dim, torch.tensor(remove_channels).cuda())
        return new, residue
    return new


def get_new_conv(old_conv, remove_channels, dim):
    if dim == 0:
        new_conv = nn.Conv2d(in_channels=old_conv.in_channels,
                             out_channels=old_conv.out_channels - len(remove_channels),
                             kernel_size=old_conv.kernel_size, stride=old_conv.stride, padding=old_conv.padding,
                             dilation=old_conv.dilation, bias=old_conv.bias is not None)
        new_conv.weight.data = select_channels(old_conv.weight.data, remove_channels, dim)
        if old_conv.bias is not None:
            new_conv.bias.data = select_channels(old_conv.bias.data, remove_channels, dim)
        return new_conv
    else:
        new_conv = nn.Conv2d(in_channels=old_conv.in_channels - len(remove_channels), out_channels=old_conv.out_channels,
                             kernel_size=old_conv.kernel_size, stride=old_conv.stride, padding=old_conv.padding,
                             dilation=old_conv.dilation, bias=old_conv.bias is not None)
        new_conv.weight.data, residue = select_channels(old_conv.weight.data, remove_channels, dim)
        if old_conv.bias is not None:
            new_conv.bias.data = old_conv.bias.data
        return new_conv, residue


def get_new_norm(old_norm, remove_channels):
    new = torch.nn.BatchNorm2d(num_features=old_norm.num_features - len(remove_channels), eps=old_norm.eps,
                               momentum=old_norm.momentum, affine=old_norm.affine,
                               track_running_stats=old_norm.track_running_stats)
    new.weight.data = select_channels(old_norm.weight.data, remove_channels, 0)
    new.bias.data = select_channels(old_norm.bias.data, remove_channels, 0)

    if old_norm.track_running_stats:
        new.running_mean.data = select_channels(old_norm.running_mean.data, remove_channels, 0)
        new.running_var.data = select_channels(old_norm.running_var.data, remove_channels, 0)

    return new


def get_new_linear(old_linear, remove_channels):
    new = torch.nn.Linear(in_features=old_linear.in_features - len(remove_channels),
                          out_features=old_linear.out_features, bias=old_linear.bias is not None)
    new.weight.data, residue = select_channels(old_linear.weight.data, remove_channels, 1)
    if old_linear.bias is not None:
        new.bias.data = old_linear.bias.data
    return new
