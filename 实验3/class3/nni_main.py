# nni_main.py
import nni
from train import train_model

if __name__ == "__main__":
    params = {
        'lr': 0.001,
        'batch_size': 64,
        'optimizer': 'Adam',
        'num_conv_layers': 3,
        'dropout_rate': 0.5
    }

    optimized_params = nni.get_next_parameter()
    if optimized_params:
        params.update(optimized_params)

    acc = train_model(params)

    nni.report_final_result(acc)