import torch.optim as optim

def get_optimizer( optimizer_type, model, learning_rate):
    
    if(optimizer_type == 'ADAM'):
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif(optimizer_type == 'ADAMW'):
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    elif(optimizer_type == 'SGD'):
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"The provided argument is not valid: {optimizer_type}")
        
    print( 'For parameter ' + optimizer_type +' optimizer ' + type(optimizer).__name__ + ' will be used in training')
        
    return optimizer