from geom3d.test_train import *
import torch
import copy

def plot_training_results(chkpt_path, config_dir):
    config = read_config(config_dir)
    np.random.seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.manual_seed(config["seed"])
    config["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"
    pymodel = Pymodel.load_from_checkpoint(chkpt_path)#,map_location={"cuda:0":"cpu"})
    pymodel.freeze()
    
    #pymodel.to('cpu')

    #config["device"] = "cpu" #"cuda" if torch.cuda.is_available() else "cpu"
    
    np.random.seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    dataset = load_data(config)
    print('y_pred', pymodel(dataset[0].to(config["device"])))
    print('y_true', dataset[0].y)
    pymode_cpu = copy.deepcopy(pymodel).to('cpu')
    print('y_pred_cpu', pymode_cpu(dataset[0].to('cpu')))
    

    train_loader, val_loader, test_loader = train_val_test_split(
        dataset, config=config
    )
    %matplotlib inline
    import matplotlib.pyplot as plt
    print("pymodel device", pymodel.device)
    fig, axis = plt.subplots(1,3, figsize=(15,5))
    for id,loader in enumerate([train_loader, val_loader, test_loader]):
        axis[id].set_ylabel('y_pred')
        axis[id].set_xlabel('y_true')
        
        for x in loader:
            with torch.no_grad():
                Y_pred = pymodel(x.to(config["device"]))
            break
        axis[id].scatter( x.y.to('cpu'),Y_pred.to('cpu').detach(),)
        axis[id].plot(x.y.to('cpu'),x.y.to('cpu'))
        axis[id].set_title(['train set', 'validation set', 'test set'][id])
    plt.show()
    plt.savefig('training_results.png')
    
    return train_loader

if __name__ == "__main__":
    from argparse import ArgumentParser
    root = os.getcwd()
    chkpt_path = os.getcwd()+"/training/SchNet_target_80K_TEST_5e4lr/epoch=91-val_loss=0.56-other_metric=0.00.ckpt"
    config_dir = os.getcwd()+"/training/SchNet_target_80K_TEST_5e4lr/"
    train_loader = plot_training_results(chkpt_path, config_dir)