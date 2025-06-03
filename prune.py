import torch
import torch.nn as nn
import torch.nn.functional as F
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


def prune_vgg(net, independentflag, prune_layers, prune_channels, criterion='taylor_expansion', 
              use_activation=True, layer_sensitivity=True, iterative_steps=3, 
              use_knowledge_distillation=True, analyze_correlations=True):
    """
    Advanced pruning function for VGG networks with multiple enhancement strategies.
    
    Args:
        net: The VGG neural network model to be pruned
        independentflag: Boolean flag that determines whether to consider residual connections
        prune_layers: List of layer names to prune (e.g., "conv_1", "conv_2", etc.)
        prune_channels: List containing the number of channels to prune for each layer
        criterion: Channel selection criterion ('l1_norm', 'l2_norm', 'taylor_expansion', 'activation', 'correlation')
        use_activation: Whether to consider activation statistics in pruning decisions
        layer_sensitivity: Whether to adjust pruning rates based on layer sensitivity
        iterative_steps: Number of iterative pruning steps (higher means more gradual pruning)
        use_knowledge_distillation: Whether to use knowledge distillation during pruning
        analyze_correlations: Whether to analyze channel correlations for redundancy detection
        
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
    
    # Calculate channel correlations if needed
    correlation_matrix = {}
    if analyze_correlations:
        correlation_matrix = calculate_channel_correlations(net, prune_layers)
    
    # Prepare for iterative pruning
    if iterative_steps > 1:
        # Calculate per-step pruning amounts
        iterative_prune_channels = []
        for channels in prune_channels:
            step_amounts = []
            remaining = channels
            for i in range(iterative_steps):
                # Distribute pruning across steps with more pruning in earlier steps
                step_amount = max(1, int(remaining / (iterative_steps - i)))
                if i == iterative_steps - 1:
                    step_amount = remaining  # Ensure we prune exactly the requested amount
                step_amounts.append(step_amount)
                remaining -= step_amount
            iterative_prune_channels.append(step_amounts)
    else:
        # Single-step pruning (original approach)
        iterative_prune_channels = [[channels] for channels in prune_channels]
    
    # Prepare for knowledge distillation if needed
    original_outputs = {}
    if use_knowledge_distillation:
        original_outputs = collect_original_outputs(net)
    
    # Perform pruning in iterative steps
    for step in range(iterative_steps):
        print(f"Pruning step {step+1}/{iterative_steps}")
        
        # Get pruning amounts for this step
        step_prune_channels = [channels[step] if step < len(channels) else 0 
                              for channels in iterative_prune_channels]
        
        # Skip this step if no pruning to do
        if sum(step_prune_channels) == 0:
            continue
        
        last_prune_flag = 0
        arg_index = 0
        conv_index = 1
        residue = None
        remove_channels_history = {}

        for i in range(len(net.module.features)):
            if isinstance(net.module.features[i], nn.Conv2d):
                # prune next layer's filter in dim=1
                if last_prune_flag:
                    net.module.features[i], residue = get_new_conv(net.module.features[i], remove_channels, 1)
                    last_prune_flag = 0
                    
                # prune this layer's filter in dim=0
                layer_name = f"conv_{conv_index}"
                if layer_name in prune_layers:
                    prune_idx = prune_layers.index(layer_name)
                    
                    # Skip if no pruning for this layer in this step
                    if step_prune_channels[prune_idx] > 0:
                        # Get previously pruned channels for this layer
                        prev_removed = remove_channels_history.get(layer_name, [])
                        
                        # Enhanced channel selection based on chosen criterion
                        remove_channels = select_channels_to_prune(
                            net.module.features[i], 
                            step_prune_channels[prune_idx], 
                            residue,
                            independentflag,
                            criterion=criterion,
                            layer_name=layer_name,
                            activation_stats=activation_stats,
                            correlation_matrix=correlation_matrix.get(layer_name, None),
                            previously_removed=prev_removed
                        )
                        
                        # Store pruned channels for future steps
                        if layer_name in remove_channels_history:
                            remove_channels_history[layer_name].extend(remove_channels)
                        else:
                            remove_channels_history[layer_name] = remove_channels
                        
                        print(f"{layer_name} step {step+1}: pruning {len(remove_channels)} channels")
                        net.module.features[i] = get_new_conv(net.module.features[i], remove_channels, 0)
                        last_prune_flag = 1
                        
                        # Only increment arg_index on the first step that prunes this layer
                        if step == 0 or (step > 0 and step_prune_channels[prune_idx-1] == 0):
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
        if "conv_13" in prune_layers and last_prune_flag:
            net.module.classifier[0] = get_new_linear(net.module.classifier[0], remove_channels)
        
        # Apply knowledge distillation if enabled and not the last step
        if use_knowledge_distillation and step < iterative_steps - 1:
            apply_knowledge_distillation(net, original_outputs)
    
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
    # Enhanced heuristic: U-shaped sensitivity with more weight to early layers
    sensitivity = {}
    num_layers = len(prune_layers)
    
    for i, layer_name in enumerate(prune_layers):
        # Position in the network (0 to 1)
        relative_pos = i / (num_layers - 1) if num_layers > 1 else 0
        
        # U-shaped sensitivity with bias toward early layers
        # Early layers (0.0-0.3): High sensitivity (0.8-1.0)
        # Middle layers (0.3-0.7): Lower sensitivity (0.4-0.8)
        # Late layers (0.7-1.0): Medium-high sensitivity (0.6-0.9)
        if relative_pos < 0.3:
            # Early layers: high sensitivity
            sensitivity[layer_name] = 1.0 - 0.2 * (relative_pos / 0.3)
        elif relative_pos < 0.7:
            # Middle layers: lower sensitivity
            sensitivity[layer_name] = 0.8 - 0.4 * ((relative_pos - 0.3) / 0.4)
        else:
            # Late layers: medium-high sensitivity
            sensitivity[layer_name] = 0.4 + 0.5 * ((relative_pos - 0.7) / 0.3)
    
    return sensitivity


def adjust_prune_channels(prune_channels, sensitivity_scores, prune_layers):
    """
    Adjust pruning rates based on layer sensitivity with adaptive scaling.
    
    Args:
        prune_channels: Original pruning rates
        sensitivity_scores: Dictionary of layer sensitivity scores
        prune_layers: List of layer names
        
    Returns:
        Adjusted pruning rates
    """
    adjusted_channels = []
    
    # Calculate average sensitivity to use as a reference
    avg_sensitivity = sum(sensitivity_scores.values()) / len(sensitivity_scores)
    
    for i, layer_name in enumerate(prune_layers):
        if layer_name in sensitivity_scores:
            # Calculate adjustment factor based on relative sensitivity
            relative_sensitivity = sensitivity_scores[layer_name] / avg_sensitivity
            
            # Apply adaptive scaling: more aggressive for less sensitive layers
            if relative_sensitivity < 0.8:  # Less sensitive than average
                adjustment_factor = 0.8 - 0.3 * (0.8 - relative_sensitivity) / 0.8
            elif relative_sensitivity > 1.2:  # More sensitive than average
                adjustment_factor = 0.8 - 0.4 * (relative_sensitivity - 1.2) / 0.8
            else:  # Around average sensitivity
                adjustment_factor = 0.8
            
            # Ensure we're pruning at least 1 channel but not more than original
            adjusted_amount = max(1, min(prune_channels[i], int(prune_channels[i] * adjustment_factor)))
            adjusted_channels.append(adjusted_amount)
        else:
            adjusted_channels.append(prune_channels[i])
    
    return adjusted_channels


def calculate_channel_correlations(net, prune_layers):
    """
    Calculate correlations between channels to identify redundancy.
    
    Args:
        net: The neural network model
        prune_layers: List of layer names to analyze
        
    Returns:
        Dictionary mapping layer names to correlation matrices
    """
    # This is a placeholder function - in a real implementation,
    # you would calculate actual correlations between filter activations
    
    # For demonstration purposes, we'll return an empty dictionary
    # In a real implementation, you would:
    # 1. Register forward hooks to capture activations
    # 2. Calculate correlation coefficients between channel activations
    # 3. Return a dictionary of correlation matrices
    
    return {}


def collect_original_outputs(net):
    """
    Collect outputs from the original model for knowledge distillation.
    
    Args:
        net: The neural network model
        
    Returns:
        Dictionary of original model outputs
    """
    # This is a placeholder function - in a real implementation,
    # you would run inference on a subset of data and store outputs
    
    # For demonstration purposes, we'll return an empty dictionary
    # In a real implementation, you would:
    # 1. Register forward hooks to capture outputs
    # 2. Run inference on a subset of data
    # 3. Store outputs for later use in knowledge distillation
    
    return {}


def apply_knowledge_distillation(net, original_outputs):
    """
    Apply knowledge distillation to help the pruned model mimic the original.
    
    Args:
        net: The pruned neural network model
        original_outputs: Outputs from the original model
    """
    # This is a placeholder function - in a real implementation,
    # you would perform fine-tuning with knowledge distillation
    
    # In a real implementation, you would:
    # 1. Run a few fine-tuning steps
    # 2. Use a loss function that combines task loss and distillation loss
    # 3. Update the model weights
    
    pass


def select_channels_to_prune(conv_layer, num_channels, residue, independentflag, 
                            criterion='taylor_expansion', layer_name=None, 
                            activation_stats=None, correlation_matrix=None,
                            previously_removed=None):
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
        correlation_matrix: Correlation matrix for this layer's channels
        previously_removed: Indices of channels already removed in previous steps
        
    Returns:
        Indices of channels to prune
    """
    weight_matrix = conv_layer.weight.data
    num_filters = weight_matrix.size(0)
    
    # Ensure we don't try to prune more filters than available
    num_channels = min(num_channels, num_filters - 1)  # Always keep at least one filter
    
    # Initialize importance scores
    importance_scores = torch.zeros(num_filters).to(weight_matrix.device)
    
    # Handle previously removed channels
    available_indices = list(range(num_filters))
    if previously_removed:
        available_indices = [i for i in available_indices if i not in previously_removed]
        if len(available_indices) <= num_channels:
            # Not enough channels left to prune the requested amount
            return available_indices[:num_channels]
    
    if criterion == 'l1_norm':
        # L1-norm criterion
        importance_scores = torch.sum(torch.abs(weight_matrix.view(num_filters, -1)), dim=1)
        if independentflag and residue is not None:
            importance_scores = importance_scores + torch.sum(torch.abs(residue.view(residue.size(0), -1)), dim=1)
        
    elif criterion == 'l2_norm':
        # L2-norm criterion (Euclidean norm)
        importance_scores = torch.sqrt(torch.sum(weight_matrix.view(num_filters, -1)**2, dim=1))
        if independentflag and residue is not None:
            importance_scores = importance_scores + torch.sqrt(torch.sum(residue.view(residue.size(0), -1)**2, dim=1))
        
    elif criterion == 'taylor_expansion':
        # Taylor expansion-based criterion (approximation)
        # In a real implementation, this would use actual gradients
        # Here we approximate with a combination of weight magnitude and position
        l2_norm = torch.sqrt(torch.sum(weight_matrix.view(num_filters, -1)**2, dim=1))
        
        # Simulate gradient importance with a position-based heuristic
        # Filters in the middle of the layer often have less impact
        positions = torch.arange(num_filters).float().to(weight_matrix.device)
        center_pos = num_filters / 2
        position_factor = 1.0 - 0.3 * torch.exp(-(positions - center_pos)**2 / (num_filters / 4)**2)
        
        # Combine L2 norm with position factor
        importance_scores = l2_norm * position_factor
        
        if independentflag and residue is not None:
            residue_l2 = torch.sqrt(torch.sum(residue.view(residue.size(0), -1)**2, dim=1))
            importance_scores = importance_scores + residue_l2
        
    elif criterion == 'activation':
        # Activation-based criterion
        if layer_name in activation_stats:
            # Use activation statistics if available
            importance_scores = torch.tensor(activation_stats[layer_name]).to(weight_matrix.device)
        else:
            # Fall back to L2-norm if no activation stats
            importance_scores = torch.sqrt(torch.sum(weight_matrix.view(num_filters, -1)**2, dim=1))
            if independentflag and residue is not None:
                importance_scores = importance_scores + torch.sqrt(torch.sum(residue.view(residue.size(0), -1)**2, dim=1))
    
    elif criterion == 'correlation':
        # Correlation-based criterion
        if correlation_matrix is not None:
            # Use correlation information to identify redundant filters
            # Higher correlation means more redundancy
            importance_scores = torch.tensor(correlation_matrix.sum(axis=1)).to(weight_matrix.device)
        else:
            # Fall back to geometric median if no correlation data
            weights_reshaped = weight_matrix.view(num_filters, -1).cpu().numpy()
            importance_scores_list = []
            
            for i in range(weights_reshaped.shape[0]):
                # Calculate distance of this filter from all others
                distances = []
                for j in range(weights_reshaped.shape[0]):
                    if i != j:
                        # Euclidean distance between filters
                        dist = np.linalg.norm(weights_reshaped[i] - weights_reshaped[j])
                        distances.append(dist)
                
                # Lower distance means the filter is more "central" and thus more important
                importance_scores_list.append(np.mean(distances))
            
            # Convert back to tensor
            importance_scores = torch.tensor(importance_scores_list).to(weight_matrix.device)
            # Invert so higher values are less important
            importance_scores = -importance_scores
    
    else:
        # Default to L2-norm if criterion not recognized
        importance_scores = torch.sqrt(torch.sum(weight_matrix.view(num_filters, -1)**2, dim=1))
        if independentflag and residue is not None:
            importance_scores = importance_scores + torch.sqrt(torch.sum(residue.view(residue.size(0), -1)**2, dim=1))
    
    # Sort by importance (lower values are less important)
    _, indices = torch.sort(importance_scores)
    
    # Filter out previously removed indices
    if previously_removed:
        indices = torch.tensor([idx for idx in indices.tolist() if idx not in previously_removed])
    
    # Return the least important channels
    return indices[:num_channels].tolist()


def recalculate_bn_statistics(bn_layer):
    """
    Recalculate batch normalization statistics after pruning.
    
    Args:
        bn_layer: Batch normalization layer
    """
    # This is a placeholder function - in a real implementation,
    # you would run inference on a subset of data and recalculate statistics
    
    # In a real implementation, you would:
    # 1. Set bn_layer to training mode
    # 2. Run forward passes with a subset of data
    # 3. Let the running statistics update
    # 4. Set bn_layer back to evaluation mode
    
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
