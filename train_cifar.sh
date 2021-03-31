
##### 2021 03 31
# CUDA_VISIBLE_DEVICES="1" nohup python -u train_cifar.py --dataset cifar100   --expname {cifar100_limber_hyper18} >log/cifar100_limber_hyper18.train &
# CUDA_VISIBLE_DEVICES="0" nohup python -u train_cifar.py --dataset cifar10   --expname {cifar10_limber_hyper18} >log/cifar10_limber_hyper18.train &

# CUDA_VISIBLE_DEVICES="0" nohup python -u train_cifar.py --dataset cifar100   --expname {cifar100_resnet18} >log/cifar100_resnet18.train &
# CUDA_VISIBLE_DEVICES="1" nohup python -u train_cifar.py --dataset cifar10   --expname {cifar10_resnet18} >log/cifar10_resnet18.train &

# CUDA_VISIBLE_DEVICES="3" nohup python -u train_cifar.py --dataset cifar100   --expname {cifar100_shared_hyper18} >log/cifar100_shared_hyper18.train &
# CUDA_VISIBLE_DEVICES="2" nohup python -u train_cifar.py --dataset cifar10   --expname {cifar10_shared_hyper18} >log/cifar10_shared_hyper18.train
