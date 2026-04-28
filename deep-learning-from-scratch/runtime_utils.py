import math


def architecture_cost(layers):
    
        
    total = 0
    for i in range(len(layers) - 1):
        total += (layers[i] + 1) * layers[i + 1]
    return total


def adaptive_iterations(reference_layers, reference_m, reference_R, target_layers, target_m):
    
  
    T_ref = architecture_cost(reference_layers)
    T_target = architecture_cost(target_layers)

    value = (reference_m * T_ref) / (target_m * T_target) * reference_R
    return min(1000, math.ceil(value))